/**
 * Sentiment Analysis Application - Frontend JavaScript
 * Handles user interactions, API calls, and result visualization
 * Updated for BERT-based sentiment analysis with detailed preprocessing steps
 */

// Global variables
let sentimentChart = null;
let selectedFile = null;

// API base URL - change this if deploying to different host
const API_BASE_URL = window.location.origin;

// DOM Elements
const textArea = document.getElementById('text-area');
const analyzeTextBtn = document.getElementById('analyze-text-btn');
const fileInput = document.getElementById('file-input');
const fileUploadArea = document.getElementById('file-upload-area');
const fileInfo = document.getElementById('file-info');
const fileName = document.getElementById('file-name');
const removeFileBtn = document.getElementById('remove-file-btn');
const analyzeFileBtn = document.getElementById('analyze-file-btn');
const loading = document.getElementById('loading');
const resultsSection = document.getElementById('results-section');
const errorMessage = document.getElementById('error-message');
const analyzeAnotherBtn = document.getElementById('analyze-another-btn');

/**
 * Initialize the application when DOM is loaded
 */
document.addEventListener('DOMContentLoaded', () => {
    initializeTabs();
    initializeEventListeners();
});

/**
 * Initialize tab switching functionality
 */
function initializeTabs() {
    const tabButtons = document.querySelectorAll('.tab-button');
    const tabContents = document.querySelectorAll('.tab-content');

    tabButtons.forEach(button => {
        button.addEventListener('click', () => {
            // Remove active class from all tabs
            tabButtons.forEach(btn => btn.classList.remove('active'));
            tabContents.forEach(content => content.classList.remove('active'));

            // Add active class to clicked tab
            button.classList.add('active');
            const tabId = button.getAttribute('data-tab');
            document.getElementById(tabId).classList.add('active');

            // Reset any previous results or errors
            hideResults();
            hideError();
        });
    });
}

/**
 * Initialize all event listeners
 */
function initializeEventListeners() {
    // Text analysis button
    analyzeTextBtn.addEventListener('click', handleTextAnalysis);

    // File upload interactions
    fileUploadArea.addEventListener('click', () => fileInput.click());
    fileInput.addEventListener('change', handleFileSelect);
    fileUploadArea.addEventListener('dragover', handleDragOver);
    fileUploadArea.addEventListener('dragleave', handleDragLeave);
    fileUploadArea.addEventListener('drop', handleFileDrop);

    // Remove file button
    removeFileBtn.addEventListener('click', removeFile);

    // Analyze file button
    analyzeFileBtn.addEventListener('click', handleFileAnalysis);

    // Analyze another button
    analyzeAnotherBtn.addEventListener('click', resetApplication);
}

/**
 * Handle text analysis from textarea
 */
async function handleTextAnalysis() {
    const text = textArea.value.trim();

    // Validate input
    if (!text) {
        showError('Please enter some text to analyze');
        return;
    }

    // Show loading and hide previous results
    showLoading();
    hideError();
    hideResults();

    try {
        // Make API call
        const response = await fetch(`${API_BASE_URL}/api/analyze`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ text: text })
        });

        const result = await response.json();

        // Hide loading
        hideLoading();

        // Check if analysis was successful
        if (result.success) {
            displayResults(result.data);
        } else {
            showError(result.error || 'An error occurred during analysis');
        }
    } catch (error) {
        hideLoading();
        showError('Failed to connect to the server. Please ensure the backend is running.');
        console.error('Error:', error);
    }
}

/**
 * Handle file selection from input
 */
function handleFileSelect(event) {
    const file = event.target.files[0];
    if (file) {
        processFile(file);
    }
}

/**
 * Handle drag over event
 */
function handleDragOver(event) {
    event.preventDefault();
    fileUploadArea.classList.add('drag-over');
}

/**
 * Handle drag leave event
 */
function handleDragLeave(event) {
    event.preventDefault();
    fileUploadArea.classList.remove('drag-over');
}

/**
 * Handle file drop event
 */
function handleFileDrop(event) {
    event.preventDefault();
    fileUploadArea.classList.remove('drag-over');

    const file = event.dataTransfer.files[0];
    if (file) {
        processFile(file);
    }
}

/**
 * Process selected file
 */
function processFile(file) {
    // Validate file type
    if (!file.name.endsWith('.txt')) {
        showError('Please select a .txt file');
        return;
    }

    // Validate file size (1MB limit)
    if (file.size > 1024 * 1024) {
        showError('File size exceeds 1MB limit');
        return;
    }

    // Store file and update UI
    selectedFile = file;
    fileName.textContent = file.name;
    fileUploadArea.classList.add('hidden');
    fileInfo.classList.remove('hidden');
    analyzeFileBtn.disabled = false;
    hideError();
}

/**
 * Remove selected file
 */
function removeFile() {
    selectedFile = null;
    fileInput.value = '';
    fileName.textContent = '';
    fileUploadArea.classList.remove('hidden');
    fileInfo.classList.add('hidden');
    analyzeFileBtn.disabled = true;
}

/**
 * Handle file analysis
 */
async function handleFileAnalysis() {
    if (!selectedFile) {
        showError('Please select a file first');
        return;
    }

    // Show loading and hide previous results
    showLoading();
    hideError();
    hideResults();

    try {
        // Create form data
        const formData = new FormData();
        formData.append('file', selectedFile);

        // Make API call
        const response = await fetch(`${API_BASE_URL}/api/analyze/file`, {
            method: 'POST',
            body: formData
        });

        const result = await response.json();

        // Hide loading
        hideLoading();

        // Check if analysis was successful
        if (result.success) {
            displayResults(result.data);
        } else {
            showError(result.error || 'An error occurred during analysis');
        }
    } catch (error) {
        hideLoading();
        showError('Failed to connect to the server. Please ensure the backend is running.');
        console.error('Error:', error);
    }
}

/**
 * Display analysis results with BERT model and detailed preprocessing steps
 */
function displayResults(data) {
    console.log('Results data:', data); // Debug log

    // Update sentiment badge with color coding
    const sentimentBadge = document.getElementById('sentiment-badge');
    sentimentBadge.textContent = data.final_sentiment.toUpperCase();
    sentimentBadge.className = `sentiment-badge ${data.final_sentiment}`;

    // Update confidence score
    const confidenceBar = document.getElementById('confidence-bar');
    const confidenceText = document.getElementById('confidence-text');
    const confidencePercent = Math.round(data.confidence * 100);
    confidenceBar.style.width = `${confidencePercent}%`;
    confidenceText.textContent = `${confidencePercent}%`;

    // Update BERT model information
    document.getElementById('model-name').textContent = data.model_info.full_name;

    // Update BERT scores for original text
    const bertOriginal = data.bert_analysis.original_text;
    document.getElementById('bert-positive').textContent = bertOriginal.probabilities.positive.toFixed(4);
    document.getElementById('bert-negative').textContent = bertOriginal.probabilities.negative.toFixed(4);
    document.getElementById('bert-neutral').textContent = bertOriginal.probabilities.neutral.toFixed(4);
    document.getElementById('bert-sentiment').textContent = bertOriginal.sentiment.toUpperCase();

    // Update BERT scores for cleaned text
    const bertCleaned = data.bert_analysis.cleaned_text;
    document.getElementById('bert-cleaned-positive').textContent = bertCleaned.probabilities.positive.toFixed(4);
    document.getElementById('bert-cleaned-negative').textContent = bertCleaned.probabilities.negative.toFixed(4);
    document.getElementById('bert-cleaned-neutral').textContent = bertCleaned.probabilities.neutral.toFixed(4);
    document.getElementById('bert-cleaned-sentiment').textContent = bertCleaned.sentiment.toUpperCase();

    // Display detailed preprocessing steps
    displayPreprocessingSteps(data.preprocessing);

    // Create sentiment visualization chart
    createSentimentChart(data.sentiment_scores);

    // Show results section
    resultsSection.classList.remove('hidden');

    // Scroll to results
    resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

/**
 * Display detailed preprocessing steps
 */
function displayPreprocessingSteps(preprocessing) {
    const steps = preprocessing.steps;

    // Step 1: Original Text
    document.getElementById('step1-text').textContent = steps['1_original'].text;
    document.getElementById('step1-count').textContent = `(${steps['1_original'].word_count} words)`;

    // Step 2: Text Cleaning
    document.getElementById('step2-text').textContent = steps['2_cleaned'].text || 'N/A';
    document.getElementById('step2-count').textContent = `(${steps['2_cleaned'].word_count} words)`;

    // Step 3: Tokenization
    const tokens3 = steps['3_tokenized'].tokens;
    document.getElementById('step3-tokens').textContent = tokens3.join(', ');
    document.getElementById('step3-count').textContent = `(${steps['3_tokenized'].token_count} tokens)`;

    // Step 4: Stopword Removal
    const step4 = steps['4_stopwords_removed'];
    document.getElementById('step4-description').textContent = step4.description;
    document.getElementById('step4-tokens').textContent = step4.tokens.join(', ');
    document.getElementById('step4-count').textContent = `(${step4.token_count} tokens)`;

    // Step 5: Lemmatization/Stemming
    const step5 = steps['5_stemmed_or_lemmatized'];
    document.getElementById('step5-description').textContent = step5.description;
    document.getElementById('step5-tokens').textContent = step5.tokens.join(', ');
    document.getElementById('step5-count').textContent = `(${step5.token_count} tokens)`;

    // Step 6: Final Cleaned Tokens (same as step 5)
    document.getElementById('step6-tokens').textContent = preprocessing.tokens.join(', ');
    document.getElementById('step6-count').textContent = `(${preprocessing.token_count} tokens)`;

    // Summary Statistics
    const originalCount = preprocessing.original_word_count;
    const finalCount = preprocessing.token_count;
    const removed = originalCount - finalCount;
    const reductionPercent = originalCount > 0 ? Math.round((removed / originalCount) * 100) : 0;

    document.getElementById('summary-original-count').textContent = originalCount;
    document.getElementById('summary-final-count').textContent = finalCount;
    document.getElementById('summary-removed-count').textContent = removed;
    document.getElementById('summary-reduction-percent').textContent = `${reductionPercent}%`;
}

/**
 * Create bar chart for sentiment visualization using Chart.js
 */
function createSentimentChart(scores) {
    const ctx = document.getElementById('sentiment-chart').getContext('2d');

    // Destroy existing chart if present
    if (sentimentChart) {
        sentimentChart.destroy();
    }

    // Create new chart
    sentimentChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: ['Positive', 'Negative', 'Neutral'],
            datasets: [{
                label: 'BERT Probability Scores',
                data: [
                    scores.positive_score,
                    scores.negative_score,
                    scores.neutral_score
                ],
                backgroundColor: [
                    'rgba(16, 185, 129, 0.8)',  // Green for positive
                    'rgba(239, 68, 68, 0.8)',   // Red for negative
                    'rgba(107, 114, 128, 0.8)'  // Gray for neutral
                ],
                borderColor: [
                    'rgba(16, 185, 129, 1)',
                    'rgba(239, 68, 68, 1)',
                    'rgba(107, 114, 128, 1)'
                ],
                borderWidth: 2
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: true,
            plugins: {
                legend: {
                    display: false
                },
                title: {
                    display: false
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    max: 1,
                    ticks: {
                        stepSize: 0.2,
                        callback: function(value) {
                            return (value * 100) + '%';
                        }
                    },
                    title: {
                        display: true,
                        text: 'Probability'
                    }
                },
                x: {
                    title: {
                        display: true,
                        text: 'Sentiment Class'
                    }
                }
            }
        }
    });
}

/**
 * Reset application to initial state
 */
function resetApplication() {
    // Clear text input
    textArea.value = '';

    // Remove file
    removeFile();

    // Hide results and errors
    hideResults();
    hideError();

    // Scroll to top
    window.scrollTo({ top: 0, behavior: 'smooth' });
}

/**
 * Show loading indicator
 */
function showLoading() {
    loading.classList.remove('hidden');
}

/**
 * Hide loading indicator
 */
function hideLoading() {
    loading.classList.add('hidden');
}

/**
 * Show results section
 */
function showResults() {
    resultsSection.classList.remove('hidden');
}

/**
 * Hide results section
 */
function hideResults() {
    resultsSection.classList.add('hidden');
}

/**
 * Show error message
 */
function showError(message) {
    errorMessage.textContent = message;
    errorMessage.classList.remove('hidden');

    // Auto-hide after 5 seconds
    setTimeout(() => {
        hideError();
    }, 5000);
}

/**
 * Hide error message
 */
function hideError() {
    errorMessage.classList.add('hidden');
}
