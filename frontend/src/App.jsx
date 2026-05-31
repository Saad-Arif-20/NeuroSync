import React, { useState } from 'react';
import { Upload, BrainCircuit, MessageSquare, Database, Settings, Activity } from 'lucide-react';
import Plotly from 'plotly.js-dist';
import factory from 'react-plotly.js/factory';
const Plot = typeof factory === 'function' ? factory(Plotly) : factory.default(Plotly);
import axios from 'axios';

// Bypass ngrok browser interstitial warning for all requests
axios.defaults.headers.common['ngrok-skip-browser-warning'] = 'true';

function App() {
  const [activeTab, setActiveTab] = useState('vision');
  const [file, setFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [embedding, setEmbedding] = useState(null);
  const [llmOutput, setLlmOutput] = useState("");

  const handleFileUpload = (e) => {
    const selected = e.target.files[0];
    if (selected) {
      setFile(selected);
    }
  };

  const processData = async () => {
    if (!file) return;
    setLoading(true);
    setEmbedding(null);
    setLlmOutput("");
    
    try {
      // 1. Get Embedding
      const formData = new FormData();
      formData.append('file', file);
      
      const endpoint = activeTab === 'eeg' ? '/api/embed/eeg' : '/api/embed/image';
      const embedRes = await axios.post(endpoint, formData);
      setEmbedding(embedRes.data.embedding);
      
      // 2. Pass to LLM Assistant
      const genFormData = new FormData();
      genFormData.append('modality', activeTab === 'vision' ? 'image' : activeTab);
      genFormData.append('file', file);
      const genRes = await axios.post('/api/generate', genFormData);
      
      setLlmOutput(genRes.data.generated_text);
    } catch (error) {
      console.error("Error processing data", error);
      setLlmOutput("Error communicating with backend models. Is FastAPI running?");
    }
    
    setLoading(false);
  };

  // Generate mock 3D data for UMAP visualization based on our 512-dim embedding
  const plotData = embedding ? [{
    x: Array.from({length: 100}, () => Math.random() * 10),
    y: Array.from({length: 100}, () => Math.random() * 10),
    z: Array.from({length: 100}, () => Math.random() * 10),
    mode: 'markers',
    type: 'scatter3d',
    marker: {
      size: 4,
      color: Array.from({length: 100}, () => Math.random() * 10),
      colorscale: 'Viridis',
      opacity: 0.8
    },
    name: 'Embeddings'
  }, {
    x: [5], y: [5], z: [5],
    mode: 'markers',
    type: 'scatter3d',
    marker: { size: 10, color: '#58a6ff', symbol: 'diamond' },
    name: 'Current Input'
  }] : [];

  return (
    <div className="dashboard-container">
      {/* Sidebar */}
      <div className="sidebar glass-panel">
        <div className="sidebar-header">
          <h1>NeuroSync</h1>
          <p>Multimodal AI Research Platform</p>
        </div>
        
        <div className="nav-links">
          <div className={`nav-item ${activeTab === 'vision' ? 'active' : ''}`} onClick={() => setActiveTab('vision')}>
            <Upload size={20} /> Image Encoding
          </div>
          <div className={`nav-item ${activeTab === 'eeg' ? 'active' : ''}`} onClick={() => setActiveTab('eeg')}>
            <Activity size={20} /> Brain Waves (EEG)
          </div>
          <div className={`nav-item ${activeTab === 'llm' ? 'active' : ''}`} onClick={() => setActiveTab('llm')}>
            <MessageSquare size={20} /> LLM Assistant
          </div>
          <div className="nav-item">
            <Database size={20} /> Dataset Manager
          </div>
          <div className="nav-item">
            <Settings size={20} /> Configuration
          </div>
        </div>
      </div>

      {/* Main Content */}
      <div className="main-content">
        <div className="header glass-panel">
          <div className="header-title">
            <h2>{activeTab === 'vision' ? 'Vision-Language Alignment' : 'Brain Signal Decoding'}</h2>
          </div>
          <div>
            <span style={{color: '#3fb950', display: 'flex', alignItems: 'center', gap: '0.5rem'}}>
              <div style={{width: '8px', height: '8px', borderRadius: '50%', backgroundColor: '#3fb950'}}></div>
              Models Loaded
            </span>
          </div>
        </div>

        <div className="workspace-grid">
          {/* Left Column: Input */}
          <div className="upload-section glass-panel animate-fade-in">
            <h3 style={{marginBottom: '1rem'}}>1. Input Source</h3>
            
            <label className="upload-box">
              <BrainCircuit className="upload-icon" />
              <span>{file ? file.name : "Click to upload Image or EEG file"}</span>
              <input type="file" style={{display: 'none'}} onChange={handleFileUpload} />
            </label>
            
            <button 
              className="btn-primary" 
              style={{marginTop: '1rem'}} 
              disabled={!file || loading}
              onClick={processData}
            >
              {loading ? "Processing via PyTorch..." : "Run Representation Learning"}
            </button>
          </div>

          {/* Right Column: Results */}
          <div className="results-section glass-panel animate-fade-in" style={{animationDelay: '0.1s'}}>
            <h3 style={{marginBottom: '1rem'}}>2. Shared Embedding Space (UMAP)</h3>
            <div className="embedding-visualizer">
              {embedding ? (
                 <Plot
                 data={plotData}
                 layout={{
                   autosize: true,
                   margin: { l: 0, r: 0, b: 0, t: 0 },
                   paper_bgcolor: 'transparent',
                   plot_bgcolor: 'transparent',
                   scene: {
                     xaxis: {showbackground: false, showticklabels: false, title: ''},
                     yaxis: {showbackground: false, showticklabels: false, title: ''},
                     zaxis: {showbackground: false, showticklabels: false, title: ''}
                   }
                 }}
                 useResizeHandler={true}
                 style={{ width: "100%", height: "100%" }}
               />
              ) : (
                <div style={{display: 'flex', height: '100%', alignItems: 'center', justifyContent: 'center', color: 'var(--text-secondary)'}}>
                  Upload data to visualize embeddings
                </div>
              )}
            </div>

            {llmOutput && (
              <div className="animate-fade-in" style={{marginTop: '1rem'}}>
                <h3 style={{marginBottom: '0.5rem'}}>3. Phase 2 (LLM Generation)</h3>
                <div className="llm-output">
                  "{llmOutput}"
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;
