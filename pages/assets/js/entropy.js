var currentSlideIndex = 0;
var imageData;
var jsonData;
var entropyData;
var chartEntropy;
var subjectToIndex = [];

// Configuration for file paths - modify these to match your SSHFS mount structure
const CONFIG = {
	// Base directory for static files - adjust this path as needed
	baseDir: 'static/',
	// Subdirectory for GIFs
	gifsDir: 'gifs/',
	// Subdirectory for PNGs (relative to subject directory)
	pngsDir: 'png/'
};

// Helper function to construct proper file paths
function getFilePath(type, filename) {
	switch (type) {
		case 'json':
			return `${CONFIG.baseDir}${filename}`;
		case 'gif':
			return `${CONFIG.baseDir}${CONFIG.gifsDir}${filename}`;
		case 'png':
			return `${CONFIG.baseDir}${CONFIG.pngsDir}${filename}`;
		default:
			return `${CONFIG.baseDir}${filename}`;
	}
}

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
	console.log('OnPage: ', onSegmentationPage(), document.URL);
	if (onSegmentationPage()) {
		imageData = jsonData[gifIndex];
		// Make sure to use the correct path construction for GIFs
		displayGIF(imageData.gif);
		if (imageData.png && imageData.png.length > 0) {
			displaySlide(0);
		} else {
			document.getElementById('pngIndex').textContent = 'No images available';
			document.getElementById('slideshowImage').src = '';
			document.getElementById('slideshowImage').alt = 'No images available';
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
		subject = data.points[0].x;
		index = subjectToIndex[subject];
		console.log('Clicked on subject: ', subject);
		changeGIF(index);
	});
}


document.addEventListener('DOMContentLoaded', () => {
	// Do not execute the code if the page is not segmentations.md
	console.log(document.URL, document.URL.includes('segmentations'));

	if (!onSegmentationPage()) {
		console.log('Not segmentations.md');
		return;
	}

	// Debug info to help troubleshoot
	console.log('Configuration:', CONFIG);

	fetch(getFilePath('json', 'gifs.json'))
		.then(response => {
			if (!response.ok) {
				throw new Error(`HTTP error! Status: ${response.status}`);
			}
			return response.json();
		})
		.then(data => {
			// Sort the data based on the name property
			data.sort((a, b) => {
				return a.name.localeCompare(b.name);
			});

			jsonData = data;
			const gifDropdown = document.getElementById('gifDropdown');

			// Clear existing options
			while (gifDropdown.firstChild) {
				gifDropdown.removeChild(gifDropdown.firstChild);
			}

			data.forEach((item, index) => {
				const option = document.createElement('option');
				option.value = index;
				option.text = item.name;
				gifDropdown.appendChild(option);
				subjectToIndex[option.text] = index;
			});

			if (data.length > 0) {
				changeGIF(0);
			} else {
				console.warn('No GIF data found in the JSON file');
			}
		})
		.catch(error => {
			console.error('Error fetching gifs.json:', error);
			// Display error message on the page
			document.getElementById('gifDropdown').innerHTML = '<option>Error loading GIFs</option>';
		});

	document.addEventListener('keydown', (event) => {
		console.log('Key pressed: ', event.key);
		if (imageData && imageData.png && imageData.png.length > 0) {
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
	});

	var chartElem = document.getElementById('gifChart');
	chartElem.addEventListener('dblclick', () => {
		console.log('Resetting zoom');
		if (chartEntropy && typeof chartEntropy.resetZoom === 'function') {
			chartEntropy.resetZoom();
		}
	});

	fetch(getFilePath('json', 'entropy.json'))
		.then(response => {
			if (!response.ok) {
				throw new Error(`HTTP error! Status: ${response.status}`);
			}
			return response.json();
		})
		.then(data => {
			entropyData = data;

			// Create a chart of the GIFs by sensitivity
			plotScatterEntropy(entropyData);
		})
		.catch(error => {
			console.error('Error fetching entropy.json:', error);
			document.getElementById('gifChart').innerHTML = '<p>Error loading entropy data</p>';
		});
});