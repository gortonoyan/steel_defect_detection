document.addEventListener('DOMContentLoaded', () => {
    const dropZone = document.getElementById('drop-zone');
    const fileInput = document.getElementById('file-input');
    const browseBtn = document.getElementById('browse-btn');
    const loading = document.getElementById('loading');
    const resultsSection = document.getElementById('results-section');
    const resultImage = document.getElementById('result-image');
    const defectList = document.getElementById('defect-list');
    const resetBtn = document.getElementById('reset-btn');

    const defectColors = {
        1: { name: 'Class 1 Defect', color: 'rgba(255, 0, 0, 0.8)' },
        2: { name: 'Class 2 Defect', color: 'rgba(0, 255, 0, 0.8)' },
        3: { name: 'Class 3 Defect', color: 'rgba(0, 0, 255, 0.8)' },
        4: { name: 'Class 4 Defect', color: 'rgba(255, 255, 0, 0.8)' }
    };

    // Event Listeners for Drag and Drop
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, preventDefaults, false);
    });

    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    ['dragenter', 'dragover'].forEach(eventName => {
        dropZone.addEventListener(eventName, () => dropZone.classList.add('dragover'), false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, () => dropZone.classList.remove('dragover'), false);
    });

    dropZone.addEventListener('drop', handleDrop, false);
    
    browseBtn.addEventListener('click', () => {
        fileInput.click();
    });

    fileInput.addEventListener('change', function() {
        if (this.files.length > 0) {
            uploadImage(this.files[0]);
        }
    });

    resetBtn.addEventListener('click', () => {
        resultsSection.classList.add('hidden');
        dropZone.classList.remove('hidden');
        fileInput.value = '';
    });

    function handleDrop(e) {
        let dt = e.dataTransfer;
        let files = dt.files;
        if (files.length > 0) {
            uploadImage(files[0]);
        }
    }

    async function uploadImage(file) {
        if (!file.type.startsWith('image/')) {
            alert('Please upload an image file.');
            return;
        }

        // Show loading state
        dropZone.classList.add('hidden');
        loading.classList.remove('hidden');

        const formData = new FormData();
        formData.append('file', file);

        try {
            const response = await fetch('/api/predict', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                const error = await response.json();
                throw new Error(error.detail || 'Failed to process image');
            }

            const data = await response.json();
            displayResults(data);
        } catch (error) {
            alert(`Error: ${error.message}`);
            loading.classList.add('hidden');
            dropZone.classList.remove('hidden');
        }
    }

    function displayResults(data) {
        loading.classList.add('hidden');
        resultsSection.classList.remove('hidden');
        
        resultImage.src = data.image_base64;
        
        // Populate Legend
        defectList.innerHTML = '';
        if (data.detected_classes.length === 0) {
            const li = document.createElement('li');
            li.className = 'defect-item';
            li.innerHTML = `<span>No defects detected ✅</span>`;
            defectList.appendChild(li);
        } else {
            data.detected_classes.forEach(cls => {
                const li = document.createElement('li');
                li.className = 'defect-item';
                li.innerHTML = `
                    <div class="color-box" style="background: ${defectColors[cls].color}"></div>
                    <span>${defectColors[cls].name}</span>
                `;
                defectList.appendChild(li);
            });
        }
    }
});
