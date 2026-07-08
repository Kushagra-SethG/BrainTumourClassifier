import { useState, useRef } from 'react';

function App() {
  const [activeTab, setActiveTab] = useState('scanner');
  const [dragActive, setDragActive] = useState(false);
  const [selectedImage, setSelectedImage] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const inputRef = useRef(null);

  const handleDrag = (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
      setDragActive(false);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFile(e.dataTransfer.files[0]);
    }
  };

  const handleChange = (e) => {
    e.preventDefault();
    if (e.target.files && e.target.files[0]) {
      handleFile(e.target.files[0]);
    }
  };

  const handleFile = (file) => {
    if (!file.type.startsWith('image/')) {
      setError('Please upload an image file (JPEG, PNG).');
      return;
    }
    setError(null);
    setSelectedImage(file);
    setPreviewUrl(URL.createObjectURL(file));
    setResult(null);
  };

  const onButtonClick = () => {
    inputRef.current.click();
  };

  const submitImage = async () => {
    if (!selectedImage) return;
    
    setLoading(true);
    setError(null);
    
    const formData = new FormData();
    formData.append('file', selectedImage);
    
    try {
      const response = await fetch('http://localhost:8000/predict', {
        method: 'POST',
        body: formData,
      });
      
      const data = await response.json();
      
      if (response.ok) {
        setResult(data);
      } else {
        setError(data.error || 'Prediction failed. Is the backend running?');
      }
    } catch (err) {
      setError('Could not connect to the server. Ensure the backend is running on port 8000.');
    } finally {
      setLoading(false);
    }
  };

  const resetAll = () => {
    setSelectedImage(null);
    setPreviewUrl(null);
    setResult(null);
    setError(null);
  };

  return (
    <div className="app-container fade-in">
      <header className="header">
        <h1>NeuroScan AI</h1>
        <p>Advanced Brain Tumor Classification via MRI</p>
      </header>

      <nav className="tabs-nav fade-in">
        <button 
          className={`tab-btn ${activeTab === 'scanner' ? 'active' : ''}`}
          onClick={() => setActiveTab('scanner')}
        >
          <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4M17 8l-5-5-5 5M12 3v12"/></svg>
          Scanner
        </button>
        <button 
          className={`tab-btn ${activeTab === 'stats' ? 'active' : ''}`}
          onClick={() => setActiveTab('stats')}
        >
          <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M12 2v20M17 5H9.5a3.5 3.5 0 0 0 0 7h5a3.5 3.5 0 0 1 0 7H6"/></svg>
          Statistics
        </button>
        <button 
          className={`tab-btn ${activeTab === 'instructions' ? 'active' : ''}`}
          onClick={() => setActiveTab('instructions')}
        >
          <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><circle cx="12" cy="12" r="10"></circle><path d="M12 16v-4"></path><path d="M12 8h.01"></path></svg>
          Instructions
        </button>
        <button 
          className={`tab-btn ${activeTab === 'about' ? 'active' : ''}`}
          onClick={() => setActiveTab('about')}
        >
          <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path><polyline points="14 2 14 8 20 8"></polyline><line x1="16" y1="13" x2="8" y2="13"></line><line x1="16" y1="17" x2="8" y2="17"></line><polyline points="10 9 9 9 8 9"></polyline></svg>
          About
        </button>
      </nav>

      <div className="card fade-in">
        {activeTab === 'scanner' && (
          <div className="fade-in">
            <h2 className="card-title">Scanner Interface</h2>
            {!previewUrl ? (
              <div 
                className={`upload-zone ${dragActive ? 'drag-active' : ''}`}
                onDragEnter={handleDrag}
                onDragLeave={handleDrag}
                onDragOver={handleDrag}
                onDrop={handleDrop}
                onClick={onButtonClick}
              >
                <div className="upload-icon">
                  <svg width="64" height="64" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1" strokeLinecap="round" strokeLinejoin="round"><rect x="3" y="3" width="18" height="18" rx="2" ry="2"/><circle cx="8.5" cy="8.5" r="1.5"/><polyline points="21 15 16 10 5 21"/></svg>
                </div>
                <p className="upload-text">Drag and drop MRI scan here</p>
                <p className="upload-text" style={{ fontSize: '0.9rem', marginTop: '0.5rem' }}>or</p>
                <button className="upload-button">Browse Files</button>
                <input 
                  ref={inputRef}
                  type="file" 
                  accept="image/*" 
                  style={{ display: 'none' }}
                  onChange={handleChange}
                />
              </div>
            ) : (
              <div className="fade-in">
                <div className="image-preview-container">
                  <img src={previewUrl} alt="MRI Preview" className="image-preview" />
                  
                  {result && (
                    <div className="result-overlay fade-in">
                      <span className="stat-label" style={{ color: 'rgba(255,255,255,0.8)' }}>AI Diagnosis</span>
                      <div className={`result-class ${result.prediction.toLowerCase()}`}>
                        {result.prediction}
                      </div>
                      <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: '0.5rem', fontSize: '0.9rem' }}>
                        <span>Confidence</span>
                        <span>{(result.confidence * 100).toFixed(2)}%</span>
                      </div>
                      <div className="confidence-bar-bg">
                        <div 
                          className={`confidence-bar-fill ${result.prediction.toLowerCase()}`} 
                          style={{ width: `${result.confidence * 100}%` }}
                        ></div>
                      </div>
                    </div>
                  )}

                  {loading && (
                    <div style={{ position: 'absolute', inset: 0, background: 'rgba(0,0,0,0.6)', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                      <div className="spinner"></div>
                    </div>
                  )}
                </div>

                {error && <div style={{ color: '#ef4444', marginBottom: '1rem', padding: '1rem', background: 'rgba(239, 68, 68, 0.1)', borderRadius: '8px' }}>{error}</div>}

                {!result && !loading && (
                  <button className="upload-button" style={{ width: '100%' }} onClick={submitImage}>
                    Analyze Scan
                  </button>
                )}

                <button className="btn-reset" onClick={resetAll} disabled={loading}>
                  {result ? 'New Analysis' : 'Cancel'}
                </button>
              </div>
            )}
          </div>
        )}

        {activeTab === 'stats' && (
          <div className="fade-in">
            <h2 className="card-title">Model Intelligence</h2>
            
            <div className="stat-item">
              <span className="stat-label">Architecture</span>
              <span className="stat-value">ResNet18 + CBAM</span>
            </div>
            
            <div className="stat-item">
              <span className="stat-label">Attention Mechanism</span>
              <span className="stat-value">Convolutional Block</span>
            </div>
            
            <div className="stat-item">
              <span className="stat-label">Detection Classes</span>
              <span className="stat-value">Normal, Tumor</span>
            </div>
            
            <div className="stat-item">
              <span className="stat-label">Input Resolution</span>
              <span className="stat-value">224 x 224 px</span>
            </div>

            <div className="stat-item" style={{ marginTop: '2rem', background: 'rgba(16, 185, 129, 0.1)', border: '1px solid rgba(16, 185, 129, 0.2)' }}>
              <span className="stat-label" style={{ color: '#10b981' }}>Validation Accuracy</span>
              <span className="stat-value" style={{ color: '#10b981' }}>99.33%</span>
            </div>
            <div className="stat-item">
              <span className="stat-label">Normal F1-Score</span>
              <span className="stat-value">0.9938</span>
            </div>
            <div className="stat-item">
              <span className="stat-label">Tumor F1-Score</span>
              <span className="stat-value">0.9927</span>
            </div>
          </div>
        )}

        {activeTab === 'instructions' && (
          <div className="fade-in content-text">
            <h2 className="card-title">How to Use NeuroScan AI</h2>
            
            <h3>1. Prepare Your MRI Scan</h3>
            <p>Ensure your MRI scan image is clear and saved in a standard image format such as JPEG, JPG, or PNG. The model is trained on axial MRI slices, so those will provide the most accurate results.</p>
            
            <h3>2. Upload the Image</h3>
            <p>Navigate to the <strong>Scanner</strong> tab. You can either drag and drop your image file directly into the designated area, or click "Browse Files" to select it from your device.</p>
            
            <h3>3. Run the Analysis</h3>
            <p>Once the image preview is visible, click the <strong>"Analyze Scan"</strong> button. The image will be securely sent to our local inference engine, which will process the scan using the ResNet18 CBAM model.</p>
            
            <h3>4. Interpret the Results</h3>
            <p>The system will overlay its findings on the image preview. You will see a prediction of either <strong>"NORMAL"</strong> or <strong>"TUMOR"</strong>, along with a confidence percentage bar representing the model's certainty.</p>
          </div>
        )}

        {activeTab === 'about' && (
          <div className="fade-in content-text">
            <h2 className="card-title">About the Project</h2>
            
            <p>NeuroScan AI is a sophisticated web application designed to demonstrate the power of deep learning in medical image analysis. It leverages a state-of-the-art Convolutional Neural Network (CNN) to detect brain tumors from MRI scans.</p>
            
            <h3>The Model</h3>
            <p>The underlying AI is a <strong>ResNet18</strong> model, significantly enhanced with a <strong>Convolutional Block Attention Module (CBAM)</strong>. CBAM allows the model to focus on the most critical features in both spatial and channel dimensions, mimicking the way a human radiologist might focus on specific anomalies in a scan. This leads to high accuracy and robust predictions.</p>
            
            <h3>Technology Stack</h3>
            <ul>
              <li><strong>Frontend:</strong> React, Vite, Vanilla CSS (Glassmorphism design)</li>
              <li><strong>Backend:</strong> FastAPI, Uvicorn</li>
              <li><strong>AI Engine:</strong> PyTorch, TorchVision</li>
            </ul>
            
            <h3>Disclaimer</h3>
            <p style={{ color: 'var(--danger)', background: 'rgba(239, 68, 68, 0.1)', padding: '1rem', borderRadius: '8px', border: '1px solid rgba(239, 68, 68, 0.3)' }}>
              This tool is for educational and demonstration purposes only. It is not an FDA-approved medical device and should not be used for actual diagnostic or clinical decision-making. Always consult a qualified medical professional for health concerns.
            </p>
          </div>
        )}
      </div>
    </div>
  );
}

export default App;
