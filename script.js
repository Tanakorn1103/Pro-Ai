function uploadImage() {
    const input = document.getElementById('imageInput');
    const previewImage = document.getElementById('previewImage');
    const animalName = document.getElementById('animalName');

    const file = input.files[0];
    if (file) {
        const reader = new FileReader();

        reader.onload = function (e) {
            previewImage.src = e.target.result;
            previewImage.style.display = 'block';

            // Convert the selected image to base64
            const base64Image = e.target.result.split(',')[1];

            // Send the base64 image to the API
            fetch('/api/cnnmodel', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    image_data: base64Image
                })
            })
            .then(response => response.json())
            .then(data => {
                animalName.innerText = `Predicted Animal: ${data.animal_type}`;
                animalName.style.display = 'block';
            });
        };

        reader.readAsDataURL(file);
    }
}
