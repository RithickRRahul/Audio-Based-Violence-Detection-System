import React from 'react';
import { AlertTriangle, CheckCircle2, Mic, Activity } from 'lucide-react';

export default function ResultsTable({ feed }) {
    if (!feed || feed.length === 0) {
        return (
            <div className="empty-state">
                <Activity size={48} style={{ opacity: 0.3 }} />
                <p>Awaiting inference data from CMAG...</p>
            </div>
        );
    }

    return (
        <div className="table-responsive">
            <table className="results-table">
                <thead>
                    <tr>
                        <th>Timestamp</th>
                        <th>System State</th>
                        <th>Raw CMAG Score</th>
                        <th>NLP Score</th>
                        <th>Final Alert Confidence</th>
                    </tr>
                </thead>
                <tbody>
                    {feed.map((result, idx) => (
                        <ResultRow key={idx} result={result} />
                    ))}
                </tbody>
            </table>
        </div>
    );
}

function ResultRow({ result }) {
    const isViolent = result.final_state === 'VIOLENCE' || result.state === 'VIOLENCE';
    const audioScore = (result.audio_score !== undefined ? (result.audio_score * 100).toFixed(1) : "0.0");
    const nlpScore = (result.nlp_score !== undefined ? (result.nlp_score * 100).toFixed(1) : "0.0");

    const finalConf = result.temporal_score !== undefined
        ? (result.temporal_score * 100).toFixed(1)
        : (result.temporal_segment_score ? (result.temporal_segment_score * 100).toFixed(1) : Math.max(audioScore, nlpScore));

    return (
        <tr className={isViolent ? 'row-violent' : 'row-safe'}>
            <td className="timestamp-cell">
                {result.timestamp}
            </td>
            <td>
                <div className={`state-badge ${isViolent ? 'state-violence' : 'state-safe'}`}>
                    {isViolent ? (
                        <><AlertTriangle size={14} /> VIOLENCE</>
                    ) : (
                        <><CheckCircle2 size={14} /> SAFE</>
                    )}
                </div>
            </td>
            <td className={`score-cell ${result.audio_score > 0.6 ? 'score-high' : ''}`}>{audioScore}%</td>
            <td className={`score-cell ${result.nlp_score > 0.6 ? 'score-high' : ''}`}>{nlpScore}%</td>
            <td className={`score-cell ${isViolent ? 'score-high' : ''}`} style={{ fontWeight: 'bold' }}>{finalConf}%</td>
        </tr>
    );
}
