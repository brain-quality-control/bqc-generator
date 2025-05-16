---
layout: single
classes: wide
title: Segmentations
permalink: /segmentations
---

# aseg.a2009+aparc segmentations

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
  <div id="gifChart"><div>
</div>

<!-- <br>
<div id="plotlyFrame">
<iframe src="volume_sig.html" style="width:100%; height:500px;"></iframe>
<div> -->

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
  transform: scale(1.25);
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
  transform: scale(1.5);
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
