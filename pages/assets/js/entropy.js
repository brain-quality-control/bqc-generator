var currentSlideIndex = 0;
var imageData;
var jsonData;
var entropyData;
var chartEntropy;
var subjectToIndex = [];

function onSegmentationPage() {
	console.log('onPage: ', document.URL);
	return document.URL.includes('segmentations');
}

function showZoomedImage(src) {
	document.getElementById('displayedGif').src = src;
	document.getElementById('displayedGif').alt = 'Zoomed view of ' + src;
}

function openFullScreen(elem) {
	if (elem.requestFullscreen) {
		elem.requestFullscreen();
	} else if (elem.webkitRequestFullscreen) { /* Safari */
		elem.webkitRequestFullscreen();
	} else if (elem.msRequestFullscreen) { /* IE11 */
		elem.msRequestFullscreen();
	}
}

function exitFullScreenMode(document) {
	if (document.exitFullscreen) {
		document.exitFullscreen();
	} else if (document.webkitExitFullscreen) { /* Safari */
		document.webkitExitFullscreen();
	} else if (document.msExitFullscreen) { /* IE11 */
		document.msExitFullscreen();
	}
}

function toggleFullScreen() {
	let elem = document.getElementById('zoomedGifContainer');
	if (!document.fullscreenElement) {
		openFullScreen(elem);
	} else {
		exitFullScreenMode(document);
	}
}

function displayGIF(gifPath) {
	console.log('Displaying GIF: ', gifPath);
	if (onSegmentationPage()) {
		document.getElementById('displayedGif').src = gifPath;
	}
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
	console.log('Changing GIF to index: ', gifIndex);
	console.log('OnPage: ', onSegmentationPage(), document.URL, onSegmentationPage);
	if (onSegmentationPage()) {
		imageData = jsonData[gifIndex];
		displayGIF(imageData.gif);
		if (imageData.png.length > 0) {
			displaySlide(0);
		} else {
			updateIndexText();
		}
	}
}

document.addEventListener('fullscreenchange', () => {
	let gif = document.getElementById('displayedGif');
	if (!document.fullscreenElement) {
		gif.classList.remove('fullscreen');
	} else {
		gif.classList.add('fullscreen');
	}
});

function extractSubjectFromGifPath(gifPath) {
	return gifPath.split('/').pop().split('.').slice(0, -1).join('.');
}

function plotScatterEntropy(data) {
	console.log('Plotting scatter entropy');
	console.log('Data: ', data);
	var trace = {
		x: data.subjects,
		y: data.std,
		mode: 'markers',
		type: 'scatter',
		name: 'Entropy',
		opacity: 0.75
	};
	var layout = {
		title: 'Entropy by subject',
		xaxis: {
			title: 'Subject'
		},
		yaxis: {
			title: 'Entropy'
		}
	};
	Plotly.newPlot('gifChart', [trace], layout);

	var entroptyChartElement = document.getElementById('gifChart');
	entroptyChartElement.on('plotly_click', function (data) {
		console.log('Clicked on subject: ', data.points);
		subject = data.points[0].x
		index = subjectToIndex[subject];
		gifpath = '/static/gifs/' + subject + '.gif'
		console.log('Clicked on subject: ', subject);
		changeGIF(index);
	}
	);
}




document.addEventListener('DOMContentLoaded', () => {
	// Do not execute the code if the page is not segmentations.md
	console.log(document.URL, document.URL.includes('segmentations'));

	if (!onSegmentationPage()) {
		console.log('Not segmentations.md');
		return;
	}

	fetch('static/gifs.json')
		.then(response => response.json())
		.then(data => {
			// Sort the data based on the gif property
			data.sort((a, b) => {
				const aBasename = extractSubjectFromGifPath(a.gif);
				const bBasename = extractSubjectFromGifPath(b.gif);
				return aBasename.localeCompare(bBasename);
			});

			jsonData = data;
			const gifDropdown = document.getElementById('gifDropdown');
			data.forEach((item, index) => {
				const basename = item["gif"].split('/').pop();
				const option = document.createElement('option');
				option.value = index;
				option.text = extractSubjectFromGifPath(item["gif"]);
				gifDropdown.appendChild(option);
				subjectToIndex[option.text] = index;
			});
			if (data.length > 0) {
				changeGIF(0);
			}
		});

	document.addEventListener('keydown', (event) => {
		console.log('Key pressed: ', event.key);
		if (event.key === 'ArrowLeft') {
			currentSlideIndex = (currentSlideIndex - 1 + imageData.png.length) % imageData.png.length;
			displaySlide(currentSlideIndex);
		} else if (event.key === 'ArrowRight') {
			currentSlideIndex = (currentSlideIndex + 1) % imageData.png.length;
			displaySlide(currentSlideIndex);
		} else if (event.key === 'Escape') {
			toggleFullScreen();
		}
	}
	);

	var chartElem = document.getElementById('gifChart');
	chartElem.addEventListener('dblclick', () => {
		console.log('Resetting zoom');
		chartEntropy.resetZoom();
	});

	fetch('static/entropy.json')
		.then(response => response.json())
		.then(data => {
			entropyData = data;

			// Create a chart of the GIFs by sensitivity
			var chartData = entropyData.subjects.map((subject, index) => {
				return { x: index, y: entropyData.std[index], gif: 'static/gifs/' + subject + '.gif' };
			});
			plotScatterEntropy(entropyData);
		});
});

