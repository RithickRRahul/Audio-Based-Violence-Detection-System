import React, { useState, useRef, useEffect } from 'react';
import { Upload, FileAudio, Loader2 } from 'lucide-react';

export default function FileUploadTile({ onResult, onUploadStart, resetTick }) {
    const [dragActive, setDragActive] = useState(false);
    const [isProcessingLocal, setIsProcessingLocal] = useState(false);
    const fileInputRef = useRef(null);

    useEffect(() => {
        setIsProcessingLocal(false);
        if (fileInputRef.current) fileInputRef.current.value = "";
    }, [resetTick]);

    const handleDrag = (e) => {
        e.preventDefault();
        e.stopPropagation();
        if (e.type === "dragenter" || e.type === "dragover") setDragActive(true);
        else if (e.type === "dragleave") setDragActive(false);
    };

    const processFile = async (file) => {
        if (!file) return;
        setIsProcessingLocal(true);

        const controller = new AbortController();
        if (onUploadStart) onUploadStart(file, controller);

        const formData = new FormData();
        formData.append("audio", file);

        try {
            const response = await fetch("http://localhost:8000/upload", {
                method: "POST",
                body: formData,
                signal: controller.signal
            });
            const data = await response.json();
            onResult(data);
        } catch (error) {
            if (error.name === 'AbortError') {
                console.log('Upload aborted by user');
            } else {
                console.error("Upload error:", error);
                alert("Failed to connect to inference backend. Is the FastAPI server running?");
            }
        } finally {
            setIsProcessingLocal(false);
            setDragActive(false);
        }
    };

    const handleDrop = (e) => {
        e.preventDefault();
        e.stopPropagation();
        setDragActive(false);
        if (e.dataTransfer.files && e.dataTransfer.files[0]) processFile(e.dataTransfer.files[0]);
    };

    const handleChange = (e) => {
        e.preventDefault();
        if (e.target.files && e.target.files[0]) processFile(e.target.files[0]);
    };

    return (
        <div className="glass-panel" style={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
            <div className="panel-header">
                <Upload size={20} className="panel-icon" />
                <h3 className="panel-title">Static Analysis</h3>
            </div>

            <div
                className={`upload-zone ${dragActive ? "drag-active" : ""}`}
                onDragEnter={handleDrag} onDragLeave={handleDrag} onDragOver={handleDrag} onDrop={handleDrop}
                onClick={() => !isProcessingLocal && fileInputRef.current?.click()}
            >
                <input ref={fileInputRef} type="file" accept="audio/*" onChange={handleChange} style={{ display: "none" }} />

                {isProcessingLocal ? (
                    <Loader2 size={48} className="upload-icon" style={{ animation: "spin 2s linear infinite" }} />
                ) : (
                    <FileAudio size={48} className="upload-icon" />
                )}

                <div className="upload-text">
                    {isProcessingLocal ? "Analyzing PyTorch Neural Graph..." : "Drag & Drop Audio File"}
                </div>
                {!isProcessingLocal && <div className="upload-hint">or click to browse local files</div>}
            </div>
        </div>
    );
}
