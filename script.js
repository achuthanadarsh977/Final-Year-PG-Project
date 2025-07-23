function predictIdentity() {
  const fileInput = document.getElementById("fileInput");
  const preview = document.getElementById("previewImage");
  const output = document.getElementById("predictionOutput");

  const file = fileInput.files[0];

  if (!file) {
    alert("Please select an image.");
    return;
  }

  const reader = new FileReader();
  reader.onload = function (e) {
    preview.src = e.target.result;
    preview.style.display = "block";

    // Simulated prediction (replace with real model logic using Flask or FastAPI)
    setTimeout(() => {
      const subjects = [113, 114, 115, 116, 117, 118, 119];
      const subject = subjects[Math.floor(Math.random() * subjects.length)];
      const confidence = (Math.random() * (0.95 - 0.85) + 0.85).toFixed(2);

      output.innerHTML = `Predicted Subject: <strong>${subject}</strong><br>Confidence: <strong>${confidence}</strong>`;
    }, 1000);
  };
  reader.readAsDataURL(file);
}
