// EmotiMap: Multi-Modal Sentiment Analysis - Main JS File

document.addEventListener('DOMContentLoaded', function() {
    // Handle file input change to show preview
    const fileInput = document.getElementById('image_file');
    const previewContainer = document.getElementById('preview-container');
    const previewImage = document.getElementById('preview-image');
    const clearImageButton = document.getElementById('clear-image-button');
    
    if (fileInput) {
        fileInput.addEventListener('change', function() {
            if (this.files && this.files[0]) {
                const file = this.files[0];
                
                // Validate file type
                const validTypes = ['image/jpeg', 'image/jpg', 'image/png'];
                if (!validTypes.includes(file.type)) {
                    showAlert('Please select a valid image file (JPEG or PNG).', 'danger');
                    fileInput.value = '';
                    previewContainer.classList.add('d-none');
                    return;
                }
                
                // Validate file size (max 16MB)
                if (file.size > 16 * 1024 * 1024) {
                    showAlert('The selected file is too large. Maximum file size is 16MB.', 'danger');
                    fileInput.value = '';
                    previewContainer.classList.add('d-none');
                    return;
                }
                
                // Show preview
                const reader = new FileReader();
                reader.onload = function(e) {
                    previewImage.src = e.target.result;
                    previewContainer.classList.remove('d-none');
                    clearImageButton.classList.remove('d-none');
                }
                reader.readAsDataURL(file);
            }
        });
    }
    
    // Clear image preview
    if (clearImageButton) {
        clearImageButton.addEventListener('click', function(e) {
            e.preventDefault();
            if (fileInput) {
                fileInput.value = '';
                previewContainer.classList.add('d-none');
                clearImageButton.classList.add('d-none');
            }
        });
    }
    
    // Form validation
    const analysisForm = document.getElementById('analysis-form');
    if (analysisForm) {
        analysisForm.addEventListener('submit', function(e) {
            const textInput = document.getElementById('text_input');
            const fileInput = document.getElementById('image_file');
            
            // Check if at least one input is provided
            if ((!textInput || textInput.value.trim() === '') && 
                (!fileInput || fileInput.files.length === 0)) {
                e.preventDefault();
                showAlert('Please provide either text or an image for analysis.', 'warning');
            }
            
            // Show loading indicator
            if (document.getElementById('loading-indicator')) {
                document.getElementById('loading-indicator').classList.remove('d-none');
            }
        });
    }
    
    // Initialize tooltips
    const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    tooltipTriggerList.map(function (tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });
    
    // Initialize popovers
    const popoverTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="popover"]'));
    popoverTriggerList.map(function (popoverTriggerEl) {
        return new bootstrap.Popover(popoverTriggerEl);
    });
    
    // Handle emotion details popover content
    const emotionDetailsButtons = document.querySelectorAll('.emotion-details-btn');
    if (emotionDetailsButtons) {
        emotionDetailsButtons.forEach(button => {
            button.addEventListener('click', function() {
                const emotions = JSON.parse(this.getAttribute('data-emotions'));
                let content = '<div class="emotion-chart-container">';
                
                // Create bar chart for emotions
                content += '<canvas id="emotionChart" width="300" height="200"></canvas>';
                content += '</div>';
                
                // Show popover
                const popover = new bootstrap.Popover(button, {
                    html: true,
                    content: content,
                    trigger: 'focus',
                    placement: 'auto'
                });
                
                popover.show();
                
                // Initialize chart after popover is shown
                setTimeout(() => {
                    const canvas = document.getElementById('emotionChart');
                    if (canvas) {
                        const ctx = canvas.getContext('2d');
                        
                        // Sort emotions from highest to lowest for better visualization
                        const sortedEmotions = Object.entries(emotions)
                            .sort((a, b) => b[1] - a[1])
                            .reduce((acc, [key, value]) => {
                                acc[key] = value;
                                return acc;
                            }, {});
                        
                        const labels = Object.keys(sortedEmotions).map(label => label.charAt(0).toUpperCase() + label.slice(1));
                        const data = Object.values(sortedEmotions);
                        
                        // Map emotion names to colors consistently
                        const emotionColors = {
                            'happy': '#28a745',    // green
                            'neutral': '#17a2b8',  // teal
                            'surprise': '#ffc107', // yellow
                            'sad': '#0d6efd',      // blue
                            'fear': '#fd7e14',     // orange
                            'disgust': '#6f42c1',  // purple
                            'angry': '#dc3545'     // red
                        };
                        
                        // Create color array in same order as the labels
                        const colors = labels.map(label => 
                            emotionColors[label.toLowerCase()] || '#6c757d'
                        );
                        
                        new Chart(ctx, {
                            type: 'bar',
                            data: {
                                labels: labels,
                                datasets: [{
                                    label: 'Emotion Scores',
                                    data: data,
                                    backgroundColor: colors,
                                    borderWidth: 1,
                                    borderColor: colors.map(color => adjustColor(color, -20))
                                }]
                            },
                            options: {
                                responsive: true,
                                maintainAspectRatio: false,
                                indexAxis: 'y',  // Horizontal bar chart for better readability
                                plugins: {
                                    legend: {
                                        display: false // Hide legend as colors are self-explanatory
                                    },
                                    tooltip: {
                                        callbacks: {
                                            label: function(context) {
                                                return `${context.parsed.x.toFixed(1)}%`;
                                            }
                                        }
                                    }
                                },
                                scales: {
                                    x: {
                                        beginAtZero: true,
                                        max: 100,
                                        title: {
                                            display: true,
                                            text: 'Confidence (%)'
                                        }
                                    }
                                }
                            }
                        });
                    }
                }, 100);
            });
        });
    }
    
    // Function to show alert messages
    function showAlert(message, type) {
        const alertsContainer = document.getElementById('alerts-container');
        if (!alertsContainer) return;
        
        const alertHtml = `
            <div class="alert alert-${type} alert-dismissible fade show" role="alert">
                ${message}
                <button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>
            </div>
        `;
        
        alertsContainer.innerHTML = alertHtml;
        
        // Auto dismiss after 5 seconds
        setTimeout(() => {
            const alert = alertsContainer.querySelector('.alert');
            if (alert) {
                const bsAlert = new bootstrap.Alert(alert);
                bsAlert.close();
            }
        }, 5000);
    }
    
    // Handle flash messages
    const flashMessages = document.querySelectorAll('.alert-flash');
    if (flashMessages.length > 0) {
        flashMessages.forEach(message => {
            setTimeout(() => {
                const bsAlert = new bootstrap.Alert(message);
                bsAlert.close();
            }, 5000);
        });
    }
    
    // Helper function to adjust color brightness
    function adjustColor(color, amount) {
        // Remove the # if present
        color = color.replace('#', '');
        
        // Parse the hex color
        let r = parseInt(color.substring(0, 2), 16);
        let g = parseInt(color.substring(2, 4), 16);
        let b = parseInt(color.substring(4, 6), 16);
        
        // Adjust each channel
        r = Math.max(0, Math.min(255, r + amount));
        g = Math.max(0, Math.min(255, g + amount));
        b = Math.max(0, Math.min(255, b + amount));
        
        // Convert back to hex
        return '#' + ((1 << 24) + (r << 16) + (g << 8) + b).toString(16).slice(1);
    }
});
