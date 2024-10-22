const canvas = document.getElementById('draw-area');
const ctx = canvas.getContext('2d');
let drawing = false;
let lastPredictionTime = 0;

// Initialize canvas with white background
ctx.fillStyle = 'white';
ctx.fillRect(0, 0, canvas.width, canvas.height);

// Update event listeners to predict during drawing
canvas.addEventListener('mousemove', (event) => {
    if (drawing) {
        draw(event);
        debouncePredict();
    }
});

canvas.addEventListener('mousedown', (event) => {
    startDrawing(event);
    debouncePredict();
});

canvas.addEventListener('mouseup', () => {
    stopDrawing();
    debouncePredict();
});

function startDrawing(event) {
  drawing = true;
  draw(event);
}

function stopDrawing() {
  drawing = false;
  ctx.beginPath();
}

function draw(event) {
  if (!drawing) return;
  ctx.lineWidth = 15;
  ctx.lineCap = 'round';
  ctx.strokeStyle = '#000';

  ctx.lineTo(event.offsetX, event.offsetY);
  ctx.stroke();
  ctx.beginPath();
  ctx.moveTo(event.offsetX, event.offsetY);
}

// Clear button should also trigger a prediction
document.getElementById('clear-btn').addEventListener('click', () => {
    ctx.fillStyle = 'white';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    debouncePredict();
});

const DEBOUNCE_DELAY = 100; // Increased from 10ms to 100ms for better performance
const MIN_PREDICTION_INTERVAL = 100; // Increased from 10ms to 100ms

let predictionTimeout = null;

function debouncePredict() {
    const now = Date.now();
    if (now - lastPredictionTime < MIN_PREDICTION_INTERVAL) {
        if (predictionTimeout) clearTimeout(predictionTimeout);
        predictionTimeout = setTimeout(() => predict(), DEBOUNCE_DELAY);
        return;
    }
    
    predict();
}

function predict() {
    lastPredictionTime = Date.now();
    
    const dataURL = canvas.toDataURL('image/png');
    fetch('/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ image: dataURL })
    })
    .then(response => response.json())
    .then(data => {
        if (!data.confidence) return;

        // Use requestAnimationFrame for smoother UI updates
        requestAnimationFrame(() => {
            updateConfidenceScores(data.confidence);
            if (data.feature_maps) showFeatureMaps(data.feature_maps);
        });
    })
    .catch(error => console.error('Prediction error:', error));
}

// Separate function for updating confidence scores
function updateConfidenceScores(confidence) {
    const scoreBars = document.querySelectorAll('.score-bar');
    scoreBars.forEach(bar => {
        const digit = bar.getAttribute('data-digit');
        const score = confidence[digit] || 0;
        const fill = bar.querySelector('.score-fill');
        
        if (fill) {
            fill.style.width = `${(score * 100).toFixed(2)}%`;
            bar.querySelector('.score-label').textContent = `${digit}: ${(score * 100).toFixed(2)}%`;
        }
    });
}

function showFeatureMaps(featureMaps) {
    for (const layer in featureMaps) {
        featureMaps[layer].forEach((imgSrc, index) => {
            const featureMapContainer = document.getElementById(`${layer}-feature-map-${index}`);
            if (!featureMapContainer) return;
            
            const img = document.createElement('img');
            img.src = `data:image/png;base64,${imgSrc}`;
            img.className = 'feature-map-image';
            
            featureMapContainer.innerHTML = '';
            featureMapContainer.appendChild(img);
        });
    }
}

function displayAverageActivations(avgActivations, layerName) {
    const activationContainer = document.getElementById('activation-container');
    const layerDiv = document.createElement('div');
    layerDiv.innerHTML = `<h4>${layerName} Average Activations:</h4>`;
    const ul = document.createElement('ul');
    ul.style.listStyle = 'none';
    ul.style.padding = '0';

    avgActivations.forEach((activation, index) => {
        const li = document.createElement('li');
        li.style.marginBottom = '5px';
        const percentage = (activation * 100).toFixed(2);
        li.innerHTML = `
            <div style="display: flex; align-items: center; gap: 10px;">
                <span>Filter ${index}:</span>
                <div style="width: 100px; height: 20px; background: #f0f0f0; border-radius: 4px; overflow: hidden;">
                    <div style="width: ${percentage}%; height: 100%; background: #4caf50;"></div>
                </div>
                <span>${percentage}%</span>
            </div>
        `;
        ul.appendChild(li);
    });

    layerDiv.appendChild(ul);
    activationContainer.innerHTML = ''; // Clear previous content
    activationContainer.appendChild(layerDiv);
}


// Remove the setInterval and use a more efficient approach
document.addEventListener('DOMContentLoaded', () => {
    // Initial prediction
    predict();
});
