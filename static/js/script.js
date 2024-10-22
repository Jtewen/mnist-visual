const canvas = document.getElementById('draw-area');
const ctx = canvas.getContext('2d');
let drawing = false;
let lastPredictionTime = 0;

// Initialize canvas with white background
ctx.fillStyle = 'white';
ctx.fillRect(0, 0, canvas.width, canvas.height);

canvas.addEventListener('mousedown', startDrawing);
canvas.addEventListener('mouseup', stopDrawing);
canvas.addEventListener('mousemove', draw);

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

document.getElementById('clear-btn').addEventListener('click', () => {
    ctx.fillStyle = 'white';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
});

function predict() {
    const currentTime = Date.now();
    if (currentTime - lastPredictionTime < 100) return;
    lastPredictionTime = currentTime;

    const dataURL = canvas.toDataURL('image/png');
    fetch('/predict', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ image: dataURL })
    })
    .then(response => response.json())
    .then(data => {
        const scoreBars = document.querySelectorAll('.score-bar');
        
        // Reset all bars
        scoreBars.forEach(bar => {
            const digit = bar.getAttribute('data-digit');
            const score = data[digit] || 0; // Get score or default to 0
            const fill = bar.querySelector('.score-fill') || document.createElement('div');
            
            if (!bar.querySelector('.score-fill')) {
                fill.classList.add('score-fill');
                bar.appendChild(fill);
            }

            fill.style.width = `${(score * 100).toFixed(2)}%`;
            bar.querySelector('.score-label').textContent = `${digit}: ${(score * 100).toFixed(2)}%`;
        });
    })
    .catch(console.error);
    showFeatureMaps();
}

function showFeatureMaps() {
    const dataURL = canvas.toDataURL('image/png');
    fetch('/feature_maps', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ image: dataURL })
    })
    .then(response => response.json())
    .then(featureMapImages => {
        const featureMapContainer = document.getElementById('feature-map-images');
        featureMapContainer.innerHTML = ''; // Clear previous images

        // Check if featureMapImages is structured correctly
        console.log('Feature Map Images:', featureMapImages);

        // Append feature maps dynamically based on layer names
        for (const layer in featureMapImages) {
            const layerDiv = document.createElement('div');
            layerDiv.innerHTML = `<h3>${layer} Feature Maps:</h3>`;
            const mapsContainer = document.createElement('div');
            mapsContainer.id = `${layer}-maps`;

            featureMapImages[layer].forEach(imgSrc => {
                const img = document.createElement('img');
                img.src = `data:image/png;base64,${imgSrc}`;
                img.style.width = '100px'; // Set width for display
                img.style.height = '100px'; // Set height for display
                img.style.margin = '5px'; // Add some margin
                mapsContainer.appendChild(img);
            });

            layerDiv.appendChild(mapsContainer);
            featureMapContainer.appendChild(layerDiv);
        }
    })
    .catch(console.error);
}

setInterval(predict, 100);
