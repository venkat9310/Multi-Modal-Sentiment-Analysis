// Main JavaScript for Sentiment Analysis App

// DOM elements
const imageUploadInput = document.getElementById('imageUpload');
const imagePreview = document.getElementById('imagePreview');
const previewContainer = document.getElementById('previewContainer');
const cameraBtn = document.getElementById('cameraBtn');
const captureBtn = document.getElementById('captureBtn');
const videoElement = document.getElementById('videoElement');
const textInput = document.getElementById('textInput');
const analyzeTextBtn = document.getElementById('analyzeTextBtn');
const analyzeImageBtn = document.getElementById('analyzeImageBtn');
const analyzeCombinedBtn = document.getElementById('analyzeCombinedBtn');
const resultsSection = document.getElementById('resultsSection');
const textResults = document.getElementById('textResults');
const imageResults = document.getElementById('imageResults');
const combinedResults = document.getElementById('combinedResults');
const loader = document.getElementById('loader');
const errorMessage = document.getElementById('errorMessage');
const resetBtn = document.getElementById('resetBtn');

// Chart objects
let textChart, faceChart, combinedChart;

// Stream variable for camera
let stream = null;

// Event listeners
document.addEventListener('DOMContentLoaded', function() {
    // Initialize charts
    initializeCharts();
    
    // Image upload preview
    imageUploadInput.addEventListener('change', handleImageUpload);
    
    // Camera functionality
    if (cameraBtn) {
        cameraBtn.addEventListener('click', toggleCamera);
    }
    
    if (captureBtn) {
        captureBtn.addEventListener('click', captureImage);
    }
    
    // Analyze buttons
    if (analyzeTextBtn) {
        analyzeTextBtn.addEventListener('click', analyzeTextOnly);
    }
    
    if (analyzeImageBtn) {
        analyzeImageBtn.addEventListener('click', analyzeImageOnly);
    }
    
    if (analyzeCombinedBtn) {
        analyzeCombinedBtn.addEventListener('click', analyzeBoth);
    }
    
    // Reset button
    if (resetBtn) {
        resetBtn.addEventListener('click', resetAnalysis);
    }
});

// Initialize empty charts
function initializeCharts() {
    // Text sentiment chart
    const textCtx = document.getElementById('textSentimentChart').getContext('2d');
    textChart = new Chart(textCtx, {
        type: 'doughnut',
        data: {
            labels: ['Positive', 'Neutral', 'Negative'],
            datasets: [{
                data: [0, 0, 0],
                backgroundColor: [
                    'rgba(40, 167, 69, 0.7)',  // success/positive
                    'rgba(255, 193, 7, 0.7)',  // warning/neutral
                    'rgba(220, 53, 69, 0.7)'   // danger/negative
                ],
                borderColor: [
                    'rgba(40, 167, 69, 1)',
                    'rgba(255, 193, 7, 1)',
                    'rgba(220, 53, 69, 1)'
                ],
                borderWidth: 1
            }]
        },
        options: {
            responsive: true,
            plugins: {
                legend: {
                    position: 'bottom',
                    labels: {
                        color: 'rgba(255, 255, 255, 0.7)'
                    }
                },
                title: {
                    display: true,
                    text: 'Text Sentiment',
                    color: 'rgba(255, 255, 255, 0.9)'
                }
            }
        }
    });
    
    // Face emotion chart
    const faceCtx = document.getElementById('faceEmotionChart').getContext('2d');
    faceChart = new Chart(faceCtx, {
        type: 'radar',
        data: {
            labels: ['Happy', 'Sad', 'Angry', 'Surprise', 'Fear', 'Disgust', 'Neutral'],
            datasets: [{
                label: 'Emotion Probability',
                data: [0, 0, 0, 0, 0, 0, 0],
                backgroundColor: 'rgba(13, 110, 253, 0.2)',
                borderColor: 'rgba(13, 110, 253, 1)',
                pointBackgroundColor: 'rgba(13, 110, 253, 1)',
                pointBorderColor: '#fff',
                pointHoverBackgroundColor: '#fff',
                pointHoverBorderColor: 'rgba(13, 110, 253, 1)'
            }]
        },
        options: {
            responsive: true,
            scales: {
                r: {
                    angleLines: {
                        color: 'rgba(255, 255, 255, 0.2)'
                    },
                    grid: {
                        color: 'rgba(255, 255, 255, 0.2)'
                    },
                    pointLabels: {
                        color: 'rgba(255, 255, 255, 0.7)'
                    },
                    ticks: {
                        backdropColor: 'transparent',
                        color: 'rgba(255, 255, 255, 0.7)'
                    }
                }
            },
            plugins: {
                legend: {
                    position: 'bottom',
                    labels: {
                        color: 'rgba(255, 255, 255, 0.7)'
                    }
                },
                title: {
                    display: true,
                    text: 'Facial Expression',
                    color: 'rgba(255, 255, 255, 0.9)'
                }
            }
        }
    });
    
    // Combined sentiment chart
    const combinedCtx = document.getElementById('combinedSentimentChart').getContext('2d');
    combinedChart = new Chart(combinedCtx, {
        type: 'bar',
        data: {
            labels: ['Text', 'Face', 'Combined'],
            datasets: [{
                label: 'Sentiment Score',
                data: [0, 0, 0],
                backgroundColor: [
                    'rgba(13, 110, 253, 0.7)',  // primary
                    'rgba(108, 117, 125, 0.7)', // secondary
                    'rgba(25, 135, 84, 0.7)'    // success
                ],
                borderColor: [
                    'rgba(13, 110, 253, 1)',
                    'rgba(108, 117, 125, 1)',
                    'rgba(25, 135, 84, 1)'
                ],
                borderWidth: 1
            }]
        },
        options: {
            responsive: true,
            scales: {
                y: {
                    beginAtZero: true,
                    min: -1,
                    max: 1,
                    grid: {
                        color: 'rgba(255, 255, 255, 0.1)'
                    },
                    ticks: {
                        color: 'rgba(255, 255, 255, 0.7)'
                    }
                },
                x: {
                    grid: {
                        color: 'rgba(255, 255, 255, 0.1)'
                    },
                    ticks: {
                        color: 'rgba(255, 255, 255, 0.7)'
                    }
                }
            },
            plugins: {
                legend: {
                    display: false
                },
                title: {
                    display: true,
                    text: 'Combined Analysis',
                    color: 'rgba(255, 255, 255, 0.9)'
                }
            }
        }
    });
}

// Handle image upload and preview
function handleImageUpload(event) {
    const file = event.target.files[0];
    if (file) {
        const reader = new FileReader();
        
        reader.onload = function(e) {
            imagePreview.src = e.target.result;
            imagePreview.style.display = 'block';
            
            // Hide video if it was active
            if (videoElement.style.display === 'block') {
                stopCamera();
            }
        };
        
        reader.readAsDataURL(file);
    }
}

// Toggle camera on/off
function toggleCamera() {
    if (videoElement.style.display === 'block') {
        stopCamera();
    } else {
        startCamera();
    }
}

// Start camera stream
function startCamera() {
    if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
        navigator.mediaDevices.getUserMedia({ video: true })
            .then(function(mediaStream) {
                stream = mediaStream;
                videoElement.srcObject = mediaStream;
                videoElement.style.display = 'block';
                imagePreview.style.display = 'none';
                captureBtn.style.display = 'block';
                videoElement.play();
            })
            .catch(function(error) {
                console.error("Could not access camera: ", error);
                showError("Could not access camera. Please ensure you have granted camera permissions.");
            });
    } else {
        showError("Your browser does not support camera access.");
    }
}

// Stop camera stream
function stopCamera() {
    if (stream) {
        stream.getTracks().forEach(function(track) {
            track.stop();
        });
        stream = null;
    }
    videoElement.style.display = 'none';
    captureBtn.style.display = 'none';
}

// Capture image from camera
function captureImage() {
    if (videoElement.style.display === 'block') {
        const canvas = document.createElement('canvas');
        canvas.width = videoElement.videoWidth;
        canvas.height = videoElement.videoHeight;
        const ctx = canvas.getContext('2d');
        ctx.drawImage(videoElement, 0, 0, canvas.width, canvas.height);
        
        // Convert to data URL
        const dataURL = canvas.toDataURL('image/jpeg');
        
        // Display the captured image
        imagePreview.src = dataURL;
        imagePreview.style.display = 'block';
        
        // Stop the camera
        stopCamera();
    }
}

// Analyze text only
function analyzeTextOnly() {
    const text = textInput.value.trim();
    
    if (!text) {
        showError("Please enter text to analyze.");
        return;
    }
    
    // Show loader
    loader.style.display = 'block';
    // Hide any previous error
    errorMessage.style.display = 'none';
    
    // Prepare form data
    const formData = new FormData();
    formData.append('text', text);
    
    // Send request to backend
    fetch('/analyze', {
        method: 'POST',
        body: formData
    })
    .then(response => {
        if (!response.ok) {
            throw new Error(`HTTP error! Status: ${response.status}`);
        }
        return response.json();
    })
    .then(data => {
        // Hide loader
        loader.style.display = 'none';
        
        // Check for errors
        if (data.error) {
            showError(data.error);
            return;
        }
        
        // Hide image and combined results
        imageResults.style.display = 'none';
        combinedResults.style.display = 'none';
        
        // Display text results
        displayTextResults(data);
    })
    .catch(error => {
        console.error('Analysis error:', error);
        loader.style.display = 'none';
        showError(`Error during text analysis: ${error.message}`);
    });
}

// Analyze image only
function analyzeImageOnly() {
    const hasImage = imagePreview.src && imagePreview.src !== '';
    
    if (!hasImage) {
        showError("Please upload or capture an image to analyze.");
        return;
    }
    
    // Show loader
    loader.style.display = 'block';
    // Hide any previous error
    errorMessage.style.display = 'none';
    
    // Prepare form data
    const formData = new FormData();
    formData.append('image', imagePreview.src);
    
    // Send request to backend
    fetch('/analyze', {
        method: 'POST',
        body: formData
    })
    .then(response => {
        if (!response.ok) {
            throw new Error(`HTTP error! Status: ${response.status}`);
        }
        return response.json();
    })
    .then(data => {
        // Hide loader
        loader.style.display = 'none';
        
        // Check for errors
        if (data.error) {
            showError(data.error);
            return;
        }
        
        // Hide text and combined results
        textResults.style.display = 'none';
        combinedResults.style.display = 'none';
        
        // Display image results
        displayImageResults(data);
    })
    .catch(error => {
        console.error('Analysis error:', error);
        loader.style.display = 'none';
        showError(`Error during image analysis: ${error.message}`);
    });
}

// Analyze both text and image
function analyzeBoth() {
    // Check if we have both text and image
    const hasText = textInput.value.trim() !== '';
    const hasImage = imagePreview.src && imagePreview.src !== '';
    
    if (!hasText && !hasImage) {
        showError("Please enter text and upload an image to analyze.");
        return;
    }
    
    if (!hasText) {
        showError("Please enter text for combined analysis.");
        return;
    }
    
    if (!hasImage) {
        showError("Please upload or capture an image for combined analysis.");
        return;
    }
    
    // Show loader
    loader.style.display = 'block';
    // Hide any previous error
    errorMessage.style.display = 'none';
    
    // Prepare form data
    const formData = new FormData();
    formData.append('text', textInput.value);
    formData.append('image', imagePreview.src);
    
    // Send request to backend
    fetch('/analyze', {
        method: 'POST',
        body: formData
    })
    .then(response => {
        if (!response.ok) {
            throw new Error(`HTTP error! Status: ${response.status}`);
        }
        return response.json();
    })
    .then(data => {
        // Hide loader
        loader.style.display = 'none';
        
        // Check for errors
        if (data.error) {
            showError(data.error);
            return;
        }
        
        // Display all results
        displayResults(data);
    })
    .catch(error => {
        console.error('Analysis error:', error);
        loader.style.display = 'none';
        showError(`Error during combined analysis: ${error.message}`);
    });
}

// Display text results only
function displayTextResults(data) {
    // Show results section
    resultsSection.style.display = 'block';
    
    // Update text sentiment chart
    if (data.text_sentiment && data.text_sentiment.score !== 0) {
        textResults.style.display = 'block';
        
        // Update chart data
        textChart.data.datasets[0].data = [
            data.text_sentiment.probabilities.positive,
            data.text_sentiment.probabilities.neutral,
            data.text_sentiment.probabilities.negative
        ];
        textChart.update();
        
        // Update text sentiment summary
        document.getElementById('textSentimentLabel').textContent = data.text_sentiment.label.charAt(0).toUpperCase() + data.text_sentiment.label.slice(1);
        document.getElementById('textSentimentScore').textContent = data.text_sentiment.score;
    }
    
    // Scroll to results
    resultsSection.scrollIntoView({ behavior: 'smooth' });
}

// Display image results only
function displayImageResults(data) {
    // Show results section
    resultsSection.style.display = 'block';
    
    // Update face emotion chart if face was detected
    if (data.face_sentiment && data.face_sentiment.emotions) {
        imageResults.style.display = 'block';
        
        // Extract emotion values in the correct order
        const emotions = [
            data.face_sentiment.emotions.happy || 0,
            data.face_sentiment.emotions.sad || 0,
            data.face_sentiment.emotions.angry || 0,
            data.face_sentiment.emotions.surprise || 0,
            data.face_sentiment.emotions.fear || 0,
            data.face_sentiment.emotions.disgust || 0,
            data.face_sentiment.emotions.neutral || 0
        ];
        
        // Update chart data
        faceChart.data.datasets[0].data = emotions;
        faceChart.update();
        
        // Update face sentiment summary
        document.getElementById('faceSentimentLabel').textContent = data.face_sentiment.label.charAt(0).toUpperCase() + data.face_sentiment.label.slice(1);
        document.getElementById('faceSentimentScore').textContent = data.face_sentiment.score;
        
        // Update primary emotion if available
        if (data.face_sentiment.primary_emotion) {
            document.getElementById('primaryEmotion').textContent = data.face_sentiment.primary_emotion.charAt(0).toUpperCase() + data.face_sentiment.primary_emotion.slice(1);
        }
    } else {
        showError("No face detected in the image.");
    }
    
    // Scroll to results
    resultsSection.scrollIntoView({ behavior: 'smooth' });
}

// Display analysis results
function displayResults(data) {
    // Show results section
    resultsSection.style.display = 'block';
    
    // Update text sentiment chart if text was provided
    if (data.text_sentiment && data.text_sentiment.score !== 0) {
        textResults.style.display = 'block';
        
        // Update chart data
        textChart.data.datasets[0].data = [
            data.text_sentiment.probabilities.positive,
            data.text_sentiment.probabilities.neutral,
            data.text_sentiment.probabilities.negative
        ];
        textChart.update();
        
        // Update text sentiment summary
        document.getElementById('textSentimentLabel').textContent = data.text_sentiment.label.charAt(0).toUpperCase() + data.text_sentiment.label.slice(1);
        document.getElementById('textSentimentScore').textContent = data.text_sentiment.score;
    } else {
        textResults.style.display = 'none';
    }
    
    // Update face emotion chart if face was detected
    if (data.face_sentiment && data.face_sentiment.emotions) {
        imageResults.style.display = 'block';
        
        // Extract emotion values in the correct order
        const emotions = [
            data.face_sentiment.emotions.happy || 0,
            data.face_sentiment.emotions.sad || 0,
            data.face_sentiment.emotions.angry || 0,
            data.face_sentiment.emotions.surprise || 0,
            data.face_sentiment.emotions.fear || 0,
            data.face_sentiment.emotions.disgust || 0,
            data.face_sentiment.emotions.neutral || 0
        ];
        
        // Update chart data
        faceChart.data.datasets[0].data = emotions;
        faceChart.update();
        
        // Update face sentiment summary
        document.getElementById('faceSentimentLabel').textContent = data.face_sentiment.label.charAt(0).toUpperCase() + data.face_sentiment.label.slice(1);
        document.getElementById('faceSentimentScore').textContent = data.face_sentiment.score;
        
        // Update primary emotion if available
        if (data.face_sentiment.primary_emotion) {
            document.getElementById('primaryEmotion').textContent = data.face_sentiment.primary_emotion.charAt(0).toUpperCase() + data.face_sentiment.primary_emotion.slice(1);
        }
    } else {
        imageResults.style.display = 'none';
    }
    
    // Update combined sentiment chart
    if (data.combined_sentiment) {
        combinedResults.style.display = 'block';
        
        // Update chart data
        combinedChart.data.datasets[0].data = [
            data.text_sentiment.score,
            data.face_sentiment.score,
            data.combined_sentiment.score
        ];
        combinedChart.update();
        
        // Update combined sentiment summary
        document.getElementById('combinedSentimentLabel').textContent = data.combined_sentiment.label.charAt(0).toUpperCase() + data.combined_sentiment.label.slice(1);
        document.getElementById('combinedSentimentScore').textContent = data.combined_sentiment.score;
    } else {
        combinedResults.style.display = 'none';
    }
    
    // Scroll to results
    resultsSection.scrollIntoView({ behavior: 'smooth' });
}

// Show error message
function showError(message) {
    errorMessage.textContent = message;
    errorMessage.style.display = 'block';
}

// Reset analysis
function resetAnalysis() {
    // Clear text input
    textInput.value = '';
    
    // Clear image preview
    imagePreview.src = '';
    imagePreview.style.display = 'none';
    
    // Stop camera if active
    if (videoElement.style.display === 'block') {
        stopCamera();
    }
    
    // Hide results
    resultsSection.style.display = 'none';
    
    // Hide error message
    errorMessage.style.display = 'none';
    
    // Reset charts to empty data
    textChart.data.datasets[0].data = [0, 0, 0];
    faceChart.data.datasets[0].data = [0, 0, 0, 0, 0, 0, 0];
    combinedChart.data.datasets[0].data = [0, 0, 0];
    
    textChart.update();
    faceChart.update();
    combinedChart.update();
}
