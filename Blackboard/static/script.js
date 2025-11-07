const BASE_URL = 'http://localhost:6060';
let currentVideoId = null;
let videoElement = null;
let startFrame = null;
let endFrame = null;
let playbackSpeed = 1.0;
let videoFPS = 30; // Default FPS, will be updated when video loads

// Initialize when page loads
document.addEventListener('DOMContentLoaded', () => {
    videoElement = document.getElementById('videoPlayer');
    setupVideoListeners();
    setupKeyboardShortcuts();
    
    // Handle file selection
    document.getElementById('videoFile').addEventListener('change', (e) => {
        const file = e.target.files[0];
        if (file) {
            document.getElementById('fileName').textContent = file.name;
            document.getElementById('uploadBtn').disabled = false;
        }
    });
});

function setupVideoListeners() {
    if (!videoElement) return;
    
    videoElement.addEventListener('loadedmetadata', () => {
        updateVideoInfo();
        // Try to get FPS from video metadata if available
        // Note: HTML5 video doesn't directly expose FPS, so we use a default
        // You might need to extract this from the video file or set it manually
    });
    
    videoElement.addEventListener('timeupdate', () => {
        updateVideoInfo();
    });

    videoElement.addEventListener('play', () => {
        updatePlayPauseButton();
    });

    videoElement.addEventListener('pause', () => {
        updatePlayPauseButton();
    });
}

function setupKeyboardShortcuts() {
    document.addEventListener('keydown', (e) => {
        // Don't trigger shortcuts when typing in inputs
        if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') {
            return;
        }
        
        switch(e.key.toLowerCase()) {
            case ' ':
                e.preventDefault();
                togglePlayPause();
                break;
            case 's':
                setStartFrame();
                break;
            case 'e':
                setEndFrame();
                break;
            case 'arrowleft':
            case 'a':
                e.preventDefault();
                seekFrame(-10);
                break;
            case 'arrowright':
            case 'd':
                e.preventDefault();
                seekFrame(10);
                break;
            case '+':
            case '=':
                e.preventDefault();
                adjustSpeed(0.25);
                break;
            case '-':
            case '_':
                e.preventDefault();
                adjustSpeed(-0.25);
                break;
        }
    });
}

function updateVideoInfo() {
    if (!videoElement || !videoElement.duration) return;
    
    const currentTime = videoElement.currentTime;
    const duration = videoElement.duration;
    
    const currentFrame = Math.floor(currentTime * videoFPS);
    const totalFrames = Math.floor(duration * videoFPS);
    
    document.getElementById('currentFrame').textContent = currentFrame;
    document.getElementById('totalFrames').textContent = totalFrames;
    document.getElementById('currentTime').textContent = formatTime(currentTime);
    document.getElementById('totalTime').textContent = formatTime(duration);
    
    // Update overlay
    document.getElementById('overlayTime').textContent = formatTime(currentTime);
    document.getElementById('overlayFrame').textContent = `Frame: ${currentFrame}`;
    
    // Update frame range inputs
    if (startFrame !== null) {
        document.getElementById('startFrame').value = startFrame;
        document.getElementById('startFrameDisplay').textContent = startFrame;
    }
    if (endFrame !== null) {
        document.getElementById('endFrame').value = endFrame;
        document.getElementById('endFrameDisplay').textContent = endFrame;
    }
}

function formatTime(seconds) {
    const hours = Math.floor(seconds / 3600);
    const mins = Math.floor((seconds % 3600) / 60);
    const secs = Math.floor(seconds % 60);
    return `${hours.toString().padStart(2, '0')}:${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
}

function seekFrame(frames) {
    if (!videoElement || !videoElement.duration) return;
    const currentTime = videoElement.currentTime;
    const newTime = Math.max(0, Math.min(videoElement.duration, currentTime + (frames / videoFPS)));
    videoElement.currentTime = newTime;
}

function togglePlayPause() {
    if (!videoElement) return;
    if (videoElement.paused) {
        videoElement.play();
    } else {
        videoElement.pause();
    }
    updatePlayPauseButton();
}

function updatePlayPauseButton() {
    const btn = document.getElementById('playPauseBtn');
    if (!btn || !videoElement) return;
    if (videoElement.paused) {
        btn.innerHTML = '▶ Play';
        btn.title = 'Space';
    } else {
        btn.innerHTML = '⏸ Pause';
        btn.title = 'Space';
    }
}

function adjustSpeed(delta) {
    playbackSpeed = Math.max(0.25, Math.min(4.0, playbackSpeed + delta));
    if (videoElement) {
        videoElement.playbackRate = playbackSpeed;
    }
    document.getElementById('playbackSpeed').textContent = playbackSpeed.toFixed(2) + 'x';
    showNotification(`Playback speed: ${playbackSpeed.toFixed(2)}x`, 'success');
}

function setStartFrame() {
    if (!videoElement || !videoElement.duration) return;
    startFrame = Math.floor(videoElement.currentTime * videoFPS);
    document.getElementById('startFrame').value = startFrame;
    document.getElementById('startFrameDisplay').textContent = startFrame;
    document.getElementById('startFrameDisplay').classList.add('highlight');
    setTimeout(() => {
        document.getElementById('startFrameDisplay').classList.remove('highlight');
    }, 1000);
    showNotification('Start frame set to ' + startFrame, 'success');
}

function setEndFrame() {
    if (!videoElement || !videoElement.duration) return;
    endFrame = Math.floor(videoElement.currentTime * videoFPS);
    document.getElementById('endFrame').value = endFrame;
    document.getElementById('endFrameDisplay').textContent = endFrame;
    document.getElementById('endFrameDisplay').classList.add('highlight');
    setTimeout(() => {
        document.getElementById('endFrameDisplay').classList.remove('highlight');
    }, 1000);
    showNotification('End frame set to ' + endFrame, 'success');
}

function useCurrentFrameRange() {
    if (!videoElement || !videoElement.duration) return;
    const currentFrame = Math.floor(videoElement.currentTime * videoFPS);
    if (startFrame === null) {
        startFrame = currentFrame;
    }
    if (endFrame === null || endFrame < startFrame) {
        endFrame = currentFrame;
    }
    document.getElementById('startFrame').value = startFrame;
    document.getElementById('endFrame').value = endFrame;
    document.getElementById('startFrameDisplay').textContent = startFrame;
    document.getElementById('endFrameDisplay').textContent = endFrame;
    showNotification(`Frame range set: ${startFrame} to ${endFrame}`, 'success');
}

async function uploadVideo() {
    const fileInput = document.getElementById('videoFile');
    const file = fileInput.files[0];
    
    if (!file) {
        showNotification('Please select a video file', 'error');
        return;
    }
    
    const uploadBtn = document.getElementById('uploadBtn');
    uploadBtn.disabled = true;
    uploadBtn.textContent = 'Uploading...';
    
    const formData = new FormData();
    formData.append('file', file);
    formData.append('filename', file.name);
    
    try {
        const response = await fetch(`${BASE_URL}/upload-video`, {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.error || 'Upload failed');
        }
        
        const result = await response.json();
        // Get video ID from input or use default
        currentVideoId = parseInt(document.getElementById('videoIdInput').value) || 3;
        
        // Load video for playback
        const videoUrl = URL.createObjectURL(file);
        videoElement.src = videoUrl;
        
        // Show video and question sections
        document.getElementById('videoSection').style.display = 'block';
        document.getElementById('questionSection').style.display = 'block';
        
        showNotification('Video uploaded successfully!', 'success');
        uploadBtn.textContent = 'Upload Video';
        uploadBtn.disabled = false;
        
    } catch (error) {
        showNotification('Error uploading video: ' + error.message, 'error');
        uploadBtn.textContent = 'Upload Video';
        uploadBtn.disabled = false;
    }
}

async function askQuestion() {
    if (!currentVideoId) {
        showNotification('Please upload a video first', 'error');
        return;
    }
    
    const questionText = document.getElementById('questionText').value.trim();
    const startFrameInput = parseInt(document.getElementById('startFrame').value);
    const endFrameInput = parseInt(document.getElementById('endFrame').value);
    
    if (!questionText) {
        showNotification('Please enter a question', 'error');
        return;
    }
    
    if (isNaN(startFrameInput) || isNaN(endFrameInput) || startFrameInput < 0 || endFrameInput < 0) {
        showNotification('Please set valid start and end frames', 'error');
        return;
    }
    
    if (startFrameInput >= endFrameInput) {
        showNotification('End frame must be greater than start frame', 'error');
        return;
    }
    
    // Disable button during request
    const askBtn = document.getElementById('askQuestionBtn');
    askBtn.disabled = true;
    askBtn.textContent = 'Processing...';
    
    // Show answer section with loading
    const answerSection = document.getElementById('answerSection');
    answerSection.style.display = 'block';
    const answerContent = document.getElementById('answerContent');
    answerContent.innerHTML = `
        <div class="loading" id="loadingIndicator">
            <div class="spinner"></div>
            <p>Processing your question...</p>
            <p class="loading-subtitle">This may take a few moments</p>
        </div>
    `;
    
    // Scroll to answer section
    answerSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
    
    try {
        // Build question with frame range if not already included
        let question = questionText;
        if (!questionText.includes('frame') && !questionText.includes('Frame')) {
            question = `${questionText} (from frame ${startFrameInput} to frame ${endFrameInput})`;
        }
        
        const response = await fetch(`${BASE_URL}/ask-question`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                question: question,
                videoid: currentVideoId
            })
        });
        
        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.error || 'Failed to get answer');
        }
        
        const answer = await response.json();
        displayAnswer(answer);
        
    } catch (error) {
        answerContent.innerHTML = `
            <div class="error-message">
                <strong>Error:</strong> ${escapeHtml(error.message)}
            </div>
        `;
    } finally {
        askBtn.disabled = false;
        askBtn.textContent = 'Ask Question';
    }
}

function displayAnswer(answer) {
    const answerContent = document.getElementById('answerContent');
    let html = '<div class="answer-section">';
    
    // Handle question class 1 (point loss analysis)
    if (answer.question_class === 1) {
        if (answer.reason) {
            html += `
                <h3>📝 Analysis</h3>
                <div class="llm-response">${escapeHtml(answer.reason)}</div>
            `;
        }
        
        if (answer.last_bounce_frame) {
            html += `
                <div class="divider"></div>
                <h3>🏓 Last Bounce Information</h3>
                <div class="bounces-section">
                    <p><strong>Last Bounce Frame:</strong> ${answer.last_bounce_frame}</p>
            `;
            
            if (answer.ball_position) {
                html += `
                    <p><strong>Ball Position:</strong> 
                        X: ${answer.ball_position.x || 'N/A'}, 
                        Y: ${answer.ball_position.y || 'N/A'}, 
                        Z: ${answer.ball_position.z || 'N/A'}
                    </p>
                `;
            }
            
            if (answer.ball_velocity) {
                html += `
                    <p><strong>Ball Velocity:</strong> 
                        VX: ${answer.ball_velocity.vx || 'N/A'}, 
                        VY: ${answer.ball_velocity.vy || 'N/A'}, 
                        VZ: ${answer.ball_velocity.vz || 'N/A'}
                    </p>
                `;
            }
            
            html += '</div>';
        }
    }
    // Handle question class 2 (trajectory analysis)
    else if (answer.question_class === 2) {
        // Extract LLM response
        let llmResponse = null;
        if (answer.analysis && answer.analysis.llmans && answer.analysis.llmans.length > 0) {
            const llmData = answer.analysis.llmans[0];
            if (typeof llmData === 'object' && llmData.response) {
                llmResponse = llmData.response;
            } else if (typeof llmData === 'string') {
                // Try to extract from string representation
                const match = llmData.match(/response="([^"]*(?:\\.[^"]*)*)"/);
                if (match) {
                    llmResponse = match[1].replace(/\\n/g, '\n').replace(/\\"/g, '"').replace(/\\\\/g, '\\');
                }
            }
        }
        
        // Display LLM response
        if (llmResponse) {
            html += `
                <h3>📝 LLM Analysis</h3>
                <div class="llm-response">${escapeHtml(llmResponse).replace(/\n/g, '<br>')}</div>
            `;
        }
        
        // Display ball bounces
        const ballBounces = answer.ball_bounces || [];
        if (ballBounces.length > 0) {
            html += `
                <div class="divider"></div>
                <h3>🏓 Ball Bounces</h3>
                <div class="bounces-section">
                    <p><strong>Total bounces:</strong> ${ballBounces.length}</p>
                    <p><strong>Bounce frames:</strong> [${ballBounces.join(', ')}]</p>
            `;
            
            // Show detailed bounce information if available
            if (answer.analysis && answer.analysis.bounces) {
                html += '<div class="bounce-detail"><strong>Detailed bounce information:</strong>';
                for (const [bounceId, bounceInfo] of Object.entries(answer.analysis.bounces)) {
                    const frame = bounceInfo.bounceFrame || bounceInfo.frame || 'N/A';
                    const segment = bounceInfo.segment || 'N/A';
                    html += `
                        <div class="bounce-item">
                            <strong>Bounce ${bounceId}:</strong> Frame ${frame} - Segment: ${segment}
                        </div>
                    `;
                }
                html += '</div>';
            }
            
            html += '</div>';
        } else {
            html += `
                <div class="divider"></div>
                <div class="warning-message">
                    ⚠️ No ball bounces found in the specified frame range
                </div>
            `;
        }
    }
    // Fallback for other response formats
    else {
        // Try to extract LLM response from various formats
        let llmResponse = null;
        if (answer.analysis && answer.analysis.llmans && answer.analysis.llmans.length > 0) {
            const llmData = answer.analysis.llmans[0];
            if (typeof llmData === 'object' && llmData.response) {
                llmResponse = llmData.response;
            } else if (typeof llmData === 'string') {
                const match = llmData.match(/response="([^"]*(?:\\.[^"]*)*)"/);
                if (match) {
                    llmResponse = match[1].replace(/\\n/g, '\n').replace(/\\"/g, '"').replace(/\\\\/g, '\\');
                }
            }
        }
        
        if (llmResponse) {
            html += `
                <h3>📝 LLM Analysis</h3>
                <div class="llm-response">${escapeHtml(llmResponse).replace(/\n/g, '<br>')}</div>
            `;
        }
        
        const ballBounces = answer.ball_bounces || [];
        if (ballBounces.length > 0) {
            html += `
                <div class="divider"></div>
                <h3>🏓 Ball Bounces</h3>
                <div class="bounces-section">
                    <p><strong>Total bounces:</strong> ${ballBounces.length}</p>
                    <p><strong>Bounce frames:</strong> [${ballBounces.join(', ')}]</p>
                </div>
            `;
        }
    }
    
    html += '</div>';
    answerContent.innerHTML = html;
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

function showNotification(message, type) {
    const statusDiv = document.getElementById('uploadStatus');
    statusDiv.textContent = message;
    statusDiv.className = `status-message ${type}`;
    setTimeout(() => {
        statusDiv.className = 'status-message';
        statusDiv.textContent = '';
    }, 5000);
}

