

document.addEventListener('DOMContentLoaded', function() {
  // Завантаження зображення та видалення
  const fileInput = document.getElementById('file');
  const imagePreview = document.getElementById('image-preview');
  const deleteButton = document.getElementById('delete-button');
  const uploadForm = document.getElementById('upload-form');
  const errorDiv = document.getElementById('error-message');


  function getResultDiv() {
      return document.getElementById('result');
  }


  if (imagePreview.src && imagePreview.src !== window.location.href && imagePreview.style.display !== 'none') {
      imagePreview.style.display = 'block';
      deleteButton.style.display = 'inline-block';
  }

  fileInput.addEventListener('change', function(){
      const file = this.files[0];

      if (file){
          const reader = new FileReader();

          reader.addEventListener('load', function(){
              imagePreview.src = reader.result;
              imagePreview.style.display = 'block';
              deleteButton.style.display = 'inline-block';
              // Ховаємо попередній результат
              const resultDiv = getResultDiv();
              if (resultDiv) {
                  resultDiv.style.display = 'none';
              }
          });

          reader.readAsDataURL(file);
      }
  });

  deleteButton.addEventListener('click', function(){
      fileInput.value = '';
      imagePreview.src = '#';
      imagePreview.style.display = 'none';
      deleteButton.style.display = 'none';

      const resultDiv = getResultDiv();
      if (resultDiv) {
          resultDiv.style.display = 'none';
      }

      if (errorDiv) {
          errorDiv.style.display = 'none';
      }
  });

  const starsContainer = document.getElementById('stars');
  for (let i = 0; i < 150; i++) {
      const star = document.createElement('div');
      star.className = 'star';
      star.style.top = Math.random() * 100 + '%';
      star.style.left = Math.random() * 100 + '%';
      star.style.width = star.style.height = Math.random() * 3 + 'px';
      star.style.animationDuration = Math.random() * 2 + 1 + 's';
      star.style.animationDelay = Math.random() * 5 + 's';
      starsContainer.appendChild(star);
  }
});
