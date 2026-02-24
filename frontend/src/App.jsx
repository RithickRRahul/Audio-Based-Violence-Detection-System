import React, { useState, useRef } from 'react';
import { Shield, Activity, RotateCcw } from 'lucide-react';
import './App.css';
import FileUploadTile from './components/FileUploadTile';
import LiveMicTile from './components/LiveMicTile';
import ResultsTable from './components/ResultsTable';

function App() {
  const [resultsFeed, setResultsFeed] = useState([]);
  const [systemState, setSystemState] = useState('SAFE');
  const [currentView, setCurrentView] = useState('INPUT'); // INPUT, PROCESSING, RESULTS
  const [audioUrl, setAudioUrl] = useState(null);
  const [resetTick, setResetTick] = useState(0);

  // Live Microphone references moved up to App level for global control
  const mediaRecorderRef = useRef(null);
  const chunksRef = useRef([]);
  const uploadControllerRef = useRef(null);
  const isCancelledRef = useRef(false); // Ref to track if we should drop the blob
  const [isLiveStreaming, setIsLiveStreaming] = useState(false);

  const handleUploadStart = (file, controller) => {
    uploadControllerRef.current = controller;
    setCurrentView('PROCESSING');
    setResultsFeed([]);
    if (audioUrl) {
      URL.revokeObjectURL(audioUrl);
    }
    setAudioUrl(URL.createObjectURL(file));
  };

  const startLiveStream = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      setIsLiveStreaming(true);
      setCurrentView('PROCESSING');
      setResultsFeed([]);
      if (audioUrl) {
        URL.revokeObjectURL(audioUrl);
        setAudioUrl(null);
      }

      chunksRef.current = [];
      isCancelledRef.current = false;
      const mediaRecorder = new MediaRecorder(stream, { mimeType: 'audio/webm' });
      mediaRecorderRef.current = mediaRecorder;

      mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          chunksRef.current.push(event.data);
        }
      };

      mediaRecorder.onstop = async () => {
        // If the user clicked "Cancel", don't upload anything.
        if (isCancelledRef.current) return;

        // User clicked "Stop & Analyze", compile and upload Blob.
        setIsLiveStreaming(false); // Transitions UI to "Analyzing..." spinner

        const webmBlob = new Blob(chunksRef.current, { type: 'audio/webm' });

        // Convert WebM to standard WAV using the browser's AudioContext to prevent FFmpeg crashes in librosa
        const audioContext = new (window.AudioContext || window.webkitAudioContext)();
        const arrayBuffer = await webmBlob.arrayBuffer();
        const audioBuffer = await audioContext.decodeAudioData(arrayBuffer);

        // Helper function to interleave raw audio data into WAV RIFF format
        const numberOfChannels = audioBuffer.numberOfChannels;
        const length = audioBuffer.length * numberOfChannels * 2;
        const buffer = new ArrayBuffer(44 + length);
        const view = new DataView(buffer);
        const channels = [];
        let offset = 0;
        let pos = 0;

        // RIFF Header
        const setString = (view, offset, string) => {
          for (let i = 0; i < string.length; i++) {
            view.setUint8(offset + i, string.charCodeAt(i));
          }
        };

        setString(view, 0, 'RIFF');
        view.setUint32(4, 36 + length, true);
        setString(view, 8, 'WAVE');
        setString(view, 12, 'fmt ');
        view.setUint32(16, 16, true);
        view.setUint16(20, 1, true); // PCM
        view.setUint16(22, numberOfChannels, true);
        view.setUint32(24, audioBuffer.sampleRate, true);
        view.setUint32(28, audioBuffer.sampleRate * 2 * numberOfChannels, true);
        view.setUint16(32, numberOfChannels * 2, true);
        view.setUint16(34, 16, true);
        setString(view, 36, 'data');
        view.setUint32(40, length, true);

        // Write PCM Samples
        for (let i = 0; i < audioBuffer.numberOfChannels; i++) {
          channels.push(audioBuffer.getChannelData(i));
        }

        offset = 44;
        while (pos < audioBuffer.length) {
          for (let i = 0; i < numberOfChannels; i++) {
            let sample = Math.max(-1, Math.min(1, channels[i][pos]));
            sample = (0.5 + sample < 0 ? sample * 32768 : sample * 32767) | 0;
            view.setInt16(offset, sample, true);
            offset += 2;
          }
          pos++;
        }

        const wavBlob = new Blob([buffer], { type: 'audio/wav' });
        const file = new File([wavBlob], "live_recording.wav", { type: 'audio/wav' });

        setAudioUrl(URL.createObjectURL(file));

        const formData = new FormData();
        formData.append("audio", file);

        const controller = new AbortController();
        uploadControllerRef.current = controller;

        try {
          const response = await fetch("http://localhost:8000/upload", {
            method: "POST",
            body: formData,
            signal: controller.signal
          });
          const data = await response.json();
          handleNewResult(data);
        } catch (error) {
          if (error.name === 'AbortError') {
            console.log('Upload aborted by user');
          } else {
            console.error("Upload error:", error);
            alert("Failed to analyze recording.");
            handleCancel();
          }
        }
      };

      mediaRecorder.start(1000); // Collect 1-second chunks locally
    } catch (err) {
      console.error("Mic access denied:", err);
      alert("Microphone access is required for live streaming.");
      handleCancel();
    }
  };

  const handleStopRecording = () => {
    if (mediaRecorderRef.current && isLiveStreaming) {
      isCancelledRef.current = false; // explicitly allow upload
      mediaRecorderRef.current.stop();
      mediaRecorderRef.current.stream.getTracks().forEach(track => track.stop());
    }
  };

  const handleCancel = () => {
    // Abort static file uploads instantly
    if (uploadControllerRef.current) {
      uploadControllerRef.current.abort();
      uploadControllerRef.current = null;
    }

    // Clean up Media Recorder if active, flag it as cancelled so onstop skips upload
    if (mediaRecorderRef.current && isLiveStreaming) {
      isCancelledRef.current = true;
      mediaRecorderRef.current.stop();
      mediaRecorderRef.current.stream.getTracks().forEach(track => track.stop());
    }

    setIsLiveStreaming(false);

    // Return to input view
    setCurrentView('INPUT');
    setResultsFeed([]);
    setSystemState('SAFE');
    if (audioUrl) {
      URL.revokeObjectURL(audioUrl);
      setAudioUrl(null);
    }
    setResetTick(r => r + 1);
  };

  const handleNewResult = (result) => {
    if (!result) return;
    setCurrentView('RESULTS');

    if (result.segments) {
      setResultsFeed(result.segments); // Isolate data, no appending
      setSystemState(result.final_state);
    } else if (result.audio_score !== undefined) {
      setResultsFeed(prev => [...prev, result]); // Live streams append dynamically
      setSystemState(result.final_state);
    }
  };

  const handleReset = () => {
    setCurrentView('INPUT');
    setResultsFeed([]);
    setSystemState('SAFE');
    if (audioUrl) {
      URL.revokeObjectURL(audioUrl);
      setAudioUrl(null);
    }
    setResetTick(r => r + 1);
  };

  return (
    <div className="app-container">
      <header className="app-header">
        <div className="brand">
          <h1 className="brand-title">
            <Shield size={36} color="var(--accent-cyan)" />
            CMAG-v2
          </h1>
          <h2 className="brand-subtitle">Hybrid Violence Detection Engine</h2>
        </div>

        <div className="system-status">
          <div className="status-badge" style={{
            background: systemState === 'VIOLENCE' ? 'rgba(239, 68, 68, 0.15)' : 'rgba(16, 185, 129, 0.1)',
            color: systemState === 'VIOLENCE' ? 'var(--accent-red)' : 'var(--accent-green)',
            borderColor: systemState === 'VIOLENCE' ? 'rgba(239, 68, 68, 0.3)' : 'rgba(16, 185, 129, 0.2)',
            boxShadow: systemState === 'VIOLENCE' ? '0 0 15px rgba(239, 68, 68, 0.2)' : '0 0 15px rgba(16, 185, 129, 0.1)'
          }}>
            <div className="status-dot" style={{
              backgroundColor: systemState === 'VIOLENCE' ? 'var(--accent-red)' : 'var(--accent-green)',
              boxShadow: systemState === 'VIOLENCE' ? '0 0 8px var(--accent-red)' : '0 0 8px var(--accent-green)',
              animation: systemState === 'VIOLENCE' ? 'pulse-red 1s infinite' : 'pulse 2s infinite'
            }}></div>
            SYSTEM {systemState}
          </div>
        </div>
      </header>

      <main className="main-layout">
        {currentView === 'INPUT' && (
          <div className="center-layout">
            <div className="centered-inputs-container">
              <FileUploadTile
                onResult={handleNewResult}
                onUploadStart={handleUploadStart}
                resetTick={resetTick}
              />
              <LiveMicTile
                onStreamStart={startLiveStream}
              />
            </div>
          </div>
        )}

        {currentView === 'PROCESSING' && (
          <div className="processing-view glass-panel" style={{ width: '100%', display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', padding: '6rem 2rem', gap: '1.5rem', position: 'relative' }}>
            <Activity size={64} className="panel-icon" style={{ animation: 'pulse-red 1.5s infinite', color: 'var(--accent-cyan)' }} />

            <h2 style={{ fontSize: '1.8rem' }}>
              {isLiveStreaming ? "Recording Live Audio..." : "Processing Audio Engine..."}
            </h2>

            <p style={{ color: 'var(--text-muted)' }}>
              {isLiveStreaming ? "Speak into the microphone. Click stop when finished to analyze." : "Analyzing acoustic transients and semantic intent via CMAG-v2."}
            </p>

            <div style={{ display: 'flex', gap: '1rem', marginTop: '2rem' }}>
              {isLiveStreaming && (
                <button
                  className="btn-primary"
                  onClick={handleStopRecording}
                  style={{ maxWidth: '300px' }}
                >
                  Stop & Analyze
                </button>
              )}

              <button
                className="btn-primary btn-danger"
                onClick={handleCancel}
                style={{ maxWidth: '300px' }}
              >
                {isLiveStreaming ? "Cancel Recording" : "Cancel Analysis"}
              </button>
            </div>
          </div>
        )}

        {currentView === 'RESULTS' && (
          <section className="feed-panel glass-panel">
            <div className="feed-header" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1.5rem' }}>
              <h2 className="feed-title">
                <Activity size={24} className="panel-icon" />
                Inference Results
              </h2>
              <button className="btn-primary" onClick={handleReset} style={{ backgroundColor: 'var(--bg-card)', border: '1px solid var(--border-subtle)', color: 'var(--text-primary)', padding: '0.6rem 1rem' }}>
                <RotateCcw size={18} /> Analyze Another File
              </button>
            </div>

            {audioUrl && (
              <div style={{ marginBottom: '1.5rem', padding: '1rem', backgroundColor: 'var(--bg-elevated)', borderRadius: '8px', border: '1px solid var(--border-subtle)' }}>
                <h4 style={{ marginBottom: '1rem', color: 'var(--text-secondary)', fontSize: '0.85rem', textTransform: 'uppercase', letterSpacing: '1px' }}>Source Audio Playback</h4>
                <audio controls src={audioUrl} style={{ width: '100%', height: '40px', outline: 'none' }} />
              </div>
            )}

            <ResultsTable feed={resultsFeed} />
          </section>
        )}
      </main>
    </div>
  );
}

export default App;
