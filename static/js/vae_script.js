const latentCanvas = document.getElementById('latent-space-canvas');
const latentCtx = latentCanvas.getContext('2d');
const generatedCanvas = document.getElementById('generated-area');
const generatedCtx = generatedCanvas.getContext('2d');
const z1Value = document.getElementById('z1-value');
const z2Value = document.getElementById('z2-value');

const Z_RANGE = 3;
let digitAverages = {};
let latentRepresentations = []; // Store latent representations for t-SNE

// Draw the latent space grid
function drawLatentSpace() {
    latentCtx.clearRect(0, 0, latentCanvas.width, latentCanvas.height);

    // Draw axes
    latentCtx.beginPath();
    latentCtx.moveTo(latentCanvas.width / 2, 0);
    latentCtx.lineTo(latentCanvas.width / 2, latentCanvas.height);
    latentCtx.moveTo(0, latentCanvas.height / 2);
    latentCtx.lineTo(latentCanvas.width, latentCanvas.height / 2);
    latentCtx.strokeStyle = '#000';
    latentCtx.lineWidth = 2;
    latentCtx.stroke();

    // Optionally, draw grid lines
    latentCtx.strokeStyle = '#e0e0e0';
    latentCtx.lineWidth = 1;
    for (let i = 1; i < 3; i++) {
        // Vertical lines
        latentCtx.beginPath();
        latentCtx.moveTo(latentCanvas.width / 2 + (latentCanvas.width / 2) * (i / 3), 0);
        latentCtx.lineTo(latentCanvas.width / 2 + (latentCanvas.width / 2) * (i / 3), latentCanvas.height);
        latentCtx.stroke();

        // Horizontal lines
        latentCtx.beginPath();
        latentCtx.moveTo(0, latentCanvas.height / 2 + (latentCanvas.height / 2) * (i / 3));
        latentCtx.lineTo(latentCanvas.width, latentCanvas.height / 2 + (latentCanvas.height / 2) * (i / 3));
        latentCtx.stroke();

        // Negative directions
        latentCtx.beginPath();
        latentCtx.moveTo(latentCanvas.width / 2 - (latentCanvas.width / 2) * (i / 3), 0);
        latentCtx.lineTo(latentCanvas.width / 2 - (latentCanvas.width / 2) * (i / 3), latentCanvas.height);
        latentCtx.stroke();

        latentCtx.beginPath();
        latentCtx.moveTo(0, latentCanvas.height / 2 - (latentCanvas.height / 2) * (i / 3));
        latentCtx.lineTo(latentCanvas.width, latentCanvas.height / 2 - (latentCanvas.height / 2) * (i / 3));
        latentCtx.stroke();
    }

    // Position digit labels based on averages
    const digitLabels = document.querySelectorAll('.digit-label');
    digitLabels.forEach(label => {
        const digit = label.getAttribute('data-digit');
        const avg = digitAverages[digit];
        if (avg) {
            const { x, y } = mapLatentToCanvas(avg[0], avg[1]);
            label.style.position = 'absolute';
            label.style.left = `${x - 10}px`; // Adjust for label width
            label.style.top = `${y + 5}px`; // Adjust for label height
        }
    });
}

// Convert z1 and z2 to canvas coordinates
function mapLatentToCanvas(z1, z2) {
    const x = (z1 / Z_RANGE) * (latentCanvas.width / 2) + latentCanvas.width / 2;
    const y = (-z2 / Z_RANGE) * (latentCanvas.height / 2) + latentCanvas.height / 2;
    return { x, y };
}

// Convert canvas coordinates to z1 and z2 values
function mapCoordinates(x, y) {
    const z1 = ((x - latentCanvas.width / 2) / (latentCanvas.width / 2)) * Z_RANGE;
    const z2 = ((latentCanvas.height / 2 - y) / (latentCanvas.height / 2)) * Z_RANGE;
    return { z1, z2 };
}

// Update the coordinates display
function updateCoordinates(z1, z2) {
    z1Value.textContent = z1.toFixed(2);
    z2Value.textContent = z2.toFixed(2);
}

// Draw a dot on the canvas at the specified coordinates
function drawDot(x, y) {
    latentCtx.clearRect(0, 0, latentCanvas.width, latentCanvas.height); // Clear the canvas
    drawLatentSpace(); // Redraw the latent space grid and labels
    latentCtx.fillStyle = 'red'; // Color of the dot
    latentCtx.beginPath();
    latentCtx.arc(x, y, 5, 0, Math.PI * 2); // Draw a circle with radius 5
    latentCtx.fill();
}

// Handle canvas click
latentCanvas.addEventListener('click', (event) => {
    const rect = latentCanvas.getBoundingClientRect();
    const x = event.clientX - rect.left;
    const y = event.clientY - rect.top;
    const { z1, z2 } = mapCoordinates(x, y);
    updateCoordinates(z1, z2);
    drawDot(x, y); // Draw the dot at the clicked position
    generateDigit(z1, z2);
});

// Generate digit based on z1 and z2
function generateDigit(z1, z2) {
    fetch('/generate', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ z1, z2 })
    })
    .then(response => response.json())
    .then(data => {
        const imgData = data.image;
        const img = new Image();
        img.onload = () => {
            generatedCtx.clearRect(0, 0, generatedCanvas.width, generatedCanvas.height);
            generatedCtx.drawImage(img, 0, 0, generatedCanvas.width, generatedCanvas.height);
        };
        img.src = imgData;
    })
    .catch(console.error);
}

// Fetch digit averages
function fetchDigitAverages() {
    fetch('/digit_averages')
        .then(response => response.json())
        .then(data => {
            if (data.error) throw new Error(data.error);
            digitAverages = data;
            drawLatentSpace();
        })
        .catch(console.error);
}

// Initial drawing after fetching averages
fetchDigitAverages();

// Automatically generate digit every 100ms
setInterval(() => {
    const z1 = parseFloat(z1Value.textContent);
    const z2 = parseFloat(z2Value.textContent);
    generateDigit(z1, z2);
}, 100);

let isDragging = false; // Track dragging state
let lastX, lastY; // Store the last mouse position

// Handle mouse down event to start dragging
latentCanvas.addEventListener('mousedown', (event) => {
    const rect = latentCanvas.getBoundingClientRect();
    lastX = event.clientX - rect.left;
    lastY = event.clientY - rect.top;
    isDragging = true; // Set dragging state to true
});

// Handle mouse move event to drag the dot
latentCanvas.addEventListener('mousemove', (event) => {
    if (!isDragging) return; // Only proceed if dragging

    const rect = latentCanvas.getBoundingClientRect();
    const x = event.clientX - rect.left;
    const y = event.clientY - rect.top;

    // Calculate the change in position
    const deltaX = x - lastX;
    const deltaY = y - lastY;

    // Update the coordinates based on the drag
    const { z1, z2 } = mapCoordinates(lastX + deltaX, lastY + deltaY);
    updateCoordinates(z1, z2);
    drawDot(lastX + deltaX, lastY + deltaY); // Draw the dot at the new position

    // Update last position
    lastX += deltaX;
    lastY += deltaY;
});

// Handle mouse up event to stop dragging
latentCanvas.addEventListener('mouseup', () => {
    isDragging = false; // Reset dragging state
});

// Handle mouse leave event to stop dragging if the mouse leaves the canvas
latentCanvas.addEventListener('mouseleave', () => {
    isDragging = false; // Reset dragging state
});
