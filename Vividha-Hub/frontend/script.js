document.addEventListener("DOMContentLoaded", () => {
  const fileInput = document.getElementById("movie-file");
  const dropZone = document.querySelector(".drop-zone");
  const convertBtn = document.querySelector(".convert-btn");
  const clearBtn = document.querySelector(".clear-btn");
  const previewBtn = document.querySelector(".preview-btn");
  const statusText = document.querySelector(".status-text");
  const progressFill = document.querySelector(".progress-fill");
  const connectionSwitch = document.getElementById("connection-switch");
  const darkSwitch = document.getElementById("dark-switch");

  // File Upload Handling
  dropZone.addEventListener("click", () => fileInput.click());

  dropZone.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropZone.style.borderColor = getComputedStyle(document.documentElement).getPropertyValue('--highlight');
  });

  dropZone.addEventListener("drop", (e) => {
    e.preventDefault();
    fileInput.files = e.dataTransfer.files;
    handleFileSelect();
  });

  fileInput.addEventListener("change", handleFileSelect);

  function handleFileSelect() {
    const file = fileInput.files[0];
    if (file) {
      dropZone.querySelector("p").textContent = file.name;
      dropZone.querySelector(".upload-icon").style.transform = "scale(1.2)";
    }
  }

  // Toggle Button Handling
  document.querySelectorAll(".toggle-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      const group = btn.closest(".toggle-group");
      group.querySelectorAll(".toggle-btn").forEach((t) => t.classList.remove("active"));
      btn.classList.add("active");
    });
  });

  // Call Python Backend for Conversion
  function callBackend() {
    if (!fileInput.files.length) {
      statusText.textContent = "Status: Please select a video file!";
      statusText.style.color = "#dc2626";
      return;
    }
    const audioEnabled = document.querySelector(".audio-yes.active") !== null;
    const subtitleEnabled = document.querySelector(".subtitle-yes.active") !== null;
    const audioLang = document.querySelector(".audio-lang").value.toLowerCase();
    const subtitleLang = document.querySelector(".subtitle-lang").value.toLowerCase();
    const inputLang = "english"; // Default for now, adjust based on video detection later
    const filePath = fileInput.files[0].path;
    const outputPath = `${filePath.split('.').slice(0, -1).join('.')}_dubbed.${filePath.split('.').pop()}`;

    const data = {
      file: filePath,
      input_lang: inputLang,
      audio_lang: audioEnabled ? audioLang : inputLang,
      subtitle_lang: subtitleEnabled ? subtitleLang : inputLang,
      output: outputPath
    };

    require('fs').writeFileSync('input.json', JSON.stringify(data));
    const { exec } = require('child_process');
    exec('python backend.py', (error) => {
      if (error) {
        statusText.textContent = "Status: Conversion Failed!";
        statusText.style.color = "#dc2626";
        return;
      }
      // Poll progress
      const interval = setInterval(() => {
        if (require('fs').existsSync('progress.json')) {
          const progress = require('fs').readFileSync('progress.json', 'utf8');
          const { percent, message } = JSON.parse(progress);
          statusText.textContent = message;
          progressFill.style.width = `${percent}%`;
          progressFill.style.color = percent === -1 ? "#dc2626" : (percent === 100 ? "#16a34a" : getComputedStyle(document.documentElement).getPropertyValue('--accent'));
          if (percent === 100 || percent === -1) clearInterval(interval);
        }
      }, 1000);
      if (require('fs').existsSync('result.json')) {
        const result = JSON.parse(require('fs').readFileSync('result.json', 'utf8'));
        if (result.status === "complete") {
          statusText.textContent = `Status: Conversion Complete! Output: ${result.output}`;
          statusText.style.color = "#16a34a";
        }
      }
    });
  }

  // Convert Button Click
  convertBtn.addEventListener("click", () => {
    callBackend();
  });

  // Preview Button Click - Simulate a preview process
  previewBtn.addEventListener("click", () => {
    simulateProcess("Preview");
  });

  // Clear Functionality
  clearBtn.addEventListener("click", () => {
    fileInput.value = "";
    dropZone.querySelector("p").textContent = "Drag & Drop Video File";
    dropZone.querySelector(".upload-icon").style.transform = "scale(1)";
    progressFill.style.width = "0%";
    statusText.textContent = "Status: Ready";
    statusText.style.color = getComputedStyle(document.documentElement).getPropertyValue('--text');
    if (require('fs').existsSync('input.json')) require('fs').unlinkSync('input.json');
    if (require('fs').existsSync('progress.json')) require('fs').unlinkSync('progress.json');
    if (require('fs').existsSync('result.json')) require('fs').unlinkSync('result.json');
  });

  // Dark Mode Toggle
  darkSwitch.addEventListener("change", () => {
    document.body.classList.toggle("dark");
    localStorage.setItem("darkMode", darkSwitch.checked);
  });

  // Load saved theme
  if (localStorage.getItem("darkMode") === "true") {
    darkSwitch.checked = true;
    document.body.classList.add("dark");
  }

  function simulateProcess(action) {
    statusText.textContent = `Status: ${action} in progress...`;
    statusText.style.color = getComputedStyle(document.documentElement).getPropertyValue('--accent');
    let progress = 0;
    progressFill.style.width = "0%";
    const interval = setInterval(() => {
      progress += 5;
      progressFill.style.width = `${progress}%`;
      if (progress >= 100) {
        clearInterval(interval);
        setTimeout(() => {
          statusText.textContent = `Status: ${action} Complete!`;
          statusText.style.color = "#16a34a";
        }, 500);
      }
    }, 100);
  }
});