var currentSlideIndex = 0;
var imageData;
var jsonData;
var entropyData;

// Chart.register(zoomPlugin);

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
            console.log(data);
            const gifDropdown = document.getElementById('gifDropdown');
            data.forEach((item, index) => {
                const basename = item["gif"].split('/').pop();
                const option = document.createElement('option');
                option.value = index;
                option.text = basename.split('.').slice(0, -1).join('.');
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

    fetch('static/entropy.json')
        .then(response => response.json())
        .then(data => {
            entropyData = data;
            console.log(entropyData);

            // Create a chart of the GIFs by sensitivity
            var chartData = entropyData.subjects.map((subject, index) => {
                return { x: index, y: entropyData.std[index], gif: 'static/gifs/' + subject + '.gif' };
            });
            var ctx = document.getElementById('gifChart').getContext('2d');
            var chart = new Chart(ctx, {
                type: 'scatter',
                data: {
                    datasets: [{
                        label: 'Subjects by entropy',
                        data: chartData,
                        pointBackgroundColor: 'blue'
                    }]
                },
                options: {
                    scales: {
                        x: {
                            beginAtZero: true,
                            ticks: {
                                callback: function (value, index, values) {
                                    return entropyData.subjects[index];
                                }
                            }
                        },
                        y: {
                            beginAtZero: true
                        }
                    },
                    onClick: function (event, elements) {
                        if (elements.length > 0) {
                            var chartElement = elements[0];
                            var gif = chart.data.datasets[chartElement.datasetIndex].data[chartElement.index].gif;
                            changeGIF(jsonData.findIndex(item => item.gif === gif));
                        }
                    },
                    plugins: {
                        zoom: {
                            zoom: {
                                wheel: {
                                    enabled: true,
                                },
                                pinch: {
                                    enabled: true
                                },
                                mode: 'x',
                            }
                        }
                    }
                }
            }
            );
        });
});