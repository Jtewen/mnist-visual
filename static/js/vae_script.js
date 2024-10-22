const latentCanvas = document.getElementById('latent-space-canvas');
const latentCtx = latentCanvas.getContext('2d');
const generatedCanvas = document.getElementById('generated-area');
const generatedCtx = generatedCanvas.getContext('2d');
const z1Value = document.getElementById('z1-value');
const z2Value = document.getElementById('z2-value');

const Z_RANGE = 3;
let digitAverages = {};

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

  // Draw grid lines
  latentCtx.strokeStyle = '#e0e0e0';
  latentCtx.lineWidth = 1;
  for (let i = 1; i < 3; i++) {
    const offset = (i / 3) * (latentCanvas.width / 2);
    
    // Vertical lines
    latentCtx.beginPath();
    latentCtx.moveTo(latentCanvas.width / 2 + offset, 0);
    latentCtx.lineTo(latentCanvas.width / 2 + offset, latentCanvas.height);
    latentCtx.stroke();

    latentCtx.beginPath();
    latentCtx.moveTo(latentCanvas.width / 2 - offset, 0);
    latentCtx.lineTo(latentCanvas.width / 2 - offset, latentCanvas.height);
    latentCtx.stroke();

    // Horizontal lines
    latentCtx.beginPath();
    latentCtx.moveTo(0, latentCanvas.height / 2 + offset);
    latentCtx.lineTo(latentCanvas.width, latentCanvas.height / 2 + offset);
    latentCtx.stroke();

    latentCtx.beginPath();
    latentCtx.moveTo(0, latentCanvas.height / 2 - offset);
    latentCtx.lineTo(latentCanvas.width, latentCanvas.height / 2 - offset);
    latentCtx.stroke();
  }

  // Position digit labels based on averages
  const digitLabels = document.querySelectorAll('.digit-label');
  digitLabels.forEach(label => {
    const digit = label.getAttribute('data-digit');
    const avg = digitAverages[digit];
    if (avg) {
      const { x, y } = mapLatentToCanvas(avg[0], avg[1]);
      label.style.left = `${x}px`;
      label.style.top = `${y}px`;
    }
  });
}

function mapLatentToCanvas(z1, z2) {
  const x = (z1 / Z_RANGE) * (latentCanvas.width / 2) + latentCanvas.width / 2;
  const y = (-z2 / Z_RANGE) * (latentCanvas.height / 2) + latentCanvas.height / 2;
  return { x, y };
}

function mapCoordinates(x, y) {
  const z1 = ((x - latentCanvas.width / 2) / (latentCanvas.width / 2)) * Z_RANGE;
  const z2 = ((latentCanvas.height / 2 - y) / (latentCanvas.height / 2)) * Z_RANGE;
  return { z1, z2 };
}

function updateCoordinates(z1, z2) {
  z1Value.textContent = z1.toFixed(2);
  z2Value.textContent = z2.toFixed(2);
}

function drawDot(x, y) {
  latentCtx.clearRect(0, 0, latentCanvas.width, latentCanvas.height);
  drawLatentSpace();
  latentCtx.fillStyle = 'red';
  latentCtx.beginPath();
  latentCtx.arc(x, y, 5, 0, Math.PI * 2);
  latentCtx.fill();
}

latentCanvas.addEventListener('click', (event) => {
  const rect = latentCanvas.getBoundingClientRect();
  const x = event.clientX - rect.left;
  const y = event.clientY - rect.top;
  const { z1, z2 } = mapCoordinates(x, y);
  updateCoordinates(z1, z2);
  drawDot(x, y);
  generateDigit(z1, z2);
});

function generateDigit(z1, z2) {
  fetch('/generate', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ z1, z2 })
  })
    .then(response => response.json())
    .then(data => {
      if (data.image) {
        const img = new Image();
        img.onload = () => {
          generatedCtx.clearRect(0, 0, generatedCanvas.width, generatedCanvas.height);
          generatedCtx.drawImage(img, 0, 0, generatedCanvas.width, generatedCanvas.height);
        };
        img.src = data.image;
      }
    })
    .catch(console.error);
}

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

fetchDigitAverages();