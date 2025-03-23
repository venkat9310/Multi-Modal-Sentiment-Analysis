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
                        const labels = Object.keys(emotions);
                        const data = Object.values(emotions);
                        
                        new Chart(ctx, {
                            type: 'bar',
                            data: {
                                labels: labels,
                                datasets: [{
                                    label: 'Emotion Scores',
                                    data: data,
                                    backgroundColor: [
                                        '#dc3545', // angry - red
                                        '#6f42c1', // disgust - purple
                                        '#fd7e14', // fear - orange
                                        '#6c757d', // sad - gray
                                        '#ffc107', // surprise - yellow
                                        '#17a2b8', // neutral - teal
                                        '#28a745'  // happy - green
                                    ],
                                    borderWidth: 1
                                }]
                            },
                            options: {
                                responsive: true,
                                maintainAspectRatio: false,
                                scales: {
                                    y: {
                                        beginAtZero: true,
                                        max: 100
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
});
