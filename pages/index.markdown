---
layout: home
title: Segmentations aparc.a2009s+aseg.mgz
---

<script>
var currentSlideIndex = 0;
var imageData;
var jsonData;

function showZoomedImage(src) {
  document.getElementById('displayedGif').src = src;
  document.getElementById('displayedGif').alt = 'Zoomed view of ' + src;
}

function toggleFullScreen() {
  let elem = document.getElementById('zoomedGifContainer');
  if (!document.fullscreenElement) {
    elem.requestFullscreen().catch(err => {
      alert(`Error attempting to enable full-screen mode: ${err.message}`);
    });
  } else {
    document.exitFullscreen();
  }
}

function displayGIF(gifPath) {
  document.getElementById('displayedGif').src = gifPath;
}

function displaySlide(index) {
  const slideshowImage = document.getElementById('slideshowImage');
  slideshowImage.src = imageData.png[index];
  slideshowImage.alt = `Slideshow Image ${index + 1}`;
  currentSlideIndex = index;
  // Update the index text
  document.getElementById('pngIndex').textContent = `Image ${index + 1} of ${imageData.png.length}`;

}

function changeGIF(gifIndex) {
  imageData = jsonData[gifIndex];
  displayGIF(imageData.gif);
  if (imageData.png.length > 0) {
    displaySlide(0);
  } else {
    updateIndexText();
  }
}

document.addEventListener('DOMContentLoaded', () => {
  fetch('static/gifs.json')
    .then(response => response.json())
    .then(data => {
      // Sort the data based on the gif property
      data.sort((a, b) => {
        const aBasename = a.gif.split('/').pop().split('.').slice(0, -1).join('.');
        const bBasename = b.gif.split('/').pop().split('.').slice(0, -1).join('.');
        return aBasename.localeCompare(bBasename);
      });

      jsonData = data;
      const gifDropdown = document.getElementById('gifDropdown');
      data.forEach((item, index) => {
        const basename = item["gif"].split('/').pop(); 
        const option = document.createElement('option');
        option.value = index;
        option.text =  basename.split('.').slice(0, -1).join('.');
        gifDropdown.appendChild(option);
      });
      if (data.length > 0) {
        changeGIF(0);
      }
    });

  document.addEventListener('keydown', (event) => {
    if (event.key === 'ArrowLeft') {
        currentSlideIndex = (currentSlideIndex - 1 + imageData.png.length) % imageData.png.length;
        displaySlide(currentSlideIndex);
      } else if (event.key === 'ArrowRight') {
        currentSlideIndex = (currentSlideIndex + 1) % imageData.png.length;
        displaySlide(currentSlideIndex);
      }
    }
    
  );
});

</script>


<style>
#gifList li {
  cursor: pointer;
  list-style-type: none;
  padding: 10px;
  background-color: #f4f4f4;
  margin-bottom: 5px;
}

#gifList li:hover {
  background-color: #ddd;
}


#zoomedGif:hover, #slideshowImage:hover {
  transform: scale(1.1);
}


#gifSelector {
  padding: 5px;
  margin: 10px 0;
  width: 200px;
}

#zoomedGifContainer{
  display: flex;
  justify-content: center;
  align-items: center;
  cursor: pointer;
}

#slideshowContainer {
  text-align: center;
  margin-bottom: 20px;
}

#displayedGif, #slideshowImage {
  transition: transform 0.25s ease;
}

#gifDropdown, #pngDropdown {
  margin-bottom: 10px;
}

#pngIndex {
  font-size: 1.2em;
  margin-bottom: 10px;
}
</style>

<div id="gifContainer">
    <select id="gifDropdown" onchange="changeGIF(this.value)">
      <!-- GIF options will be populated here -->
    </select>
    <br>
    <div id="zoomedGifContainer" onclick="toggleFullScreen()">
      <img id="displayedGif" src="" alt="Zoomed GIF" style="max-width: 100%; max-height: 100%;">
    </div>    
</div>
<br>
<div id="slideshowContainer">
  <div id="indexContainer">
    <p id="pngIndex">Image 1 of N</p>
  </div>
  <img id="slideshowImage" src="" alt="Slideshow Image" style="max-width: 100%; max-height: 100%;">
</div>