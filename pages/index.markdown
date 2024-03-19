---
layout: home
title: Segmentations aparc.a2009s+aseg.mgz
---

<head>
  <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/hammerjs@2.0.8"></script>
  <script src="https://cdn.jsdelivr.net/npm/chartjs-plugin-zoom"></script>
  <script type="text/javascript" src="assets/js/entropy.js"></script>>
</head>

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
<br>
<div id="gifChartContainer">
  <p>Subjects by entropy</p>
  <canvas id="gifChart"></canvas>
</div>




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
  transform: scale(1.5);
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

#zoomedGifContainer.fullscreen {
  transform: scale(2);
}

#slideshowContainer {
  text-align: center;
  margin-bottom: 20px;
}

#displayedGif, #slideshowImage {
  transition: transform 0.25s ease;
}

#displayedGif.fullscreen {
  transform: scale(1.25);
}


#gifDropdown, #pngDropdown {
  margin-bottom: 10px;
}

#pngIndex {
  font-size: 1.2em;
  margin-bottom: 10px;
}
</style>
