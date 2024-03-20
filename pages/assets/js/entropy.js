var currentSlideIndex = 0;
var imageData;
var jsonData;
var entropyData;
var chartEntropy;

// Chart.register(zoomPlugin);

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

document.addEventListener('DOMContentLoaded', () => {
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
            console.log(data);
            const gifDropdown = document.getElementById('gifDropdown');
            data.forEach((item, index) => {
                const basename = item["gif"].split('/').pop();
                const option = document.createElement('option');
                option.value = index;
                option.text = extractSubjectFromGifPath(item["gif"]);
                // option.text = basename.split('.').slice(0, -1).join('.');
                gifDropdown.appendChild(option);
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
        console.log(chartElem);
        chartEntropy.resetZoom();
    });


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
                            document.getElementById('gifDropdown').value = jsonData.findIndex(item => item.gif === gif);
                        }
                    },
                    plugins: {
                        zoom: {
                            pan: {
                                enabled: true,
                                mode: 'x',
                                modifierKey: ['ctrl']
                            },
                            zoom: {
                                drag: {
                                    enabled: true,
                                },
                                wheel: {
                                    enabled: true,
                                },
                                pinch: {
                                    enabled: true
                                },
                                mode: 'x',
                                onZoomComplete: function (chart) {
                                    const start = chart.chart.scales.x._startValue // This is first visible value
                                    const end = start + chart.chart.scales.x._valueRange // This is the last visible value
                                    const points = chart.chart.config.data.datasets[0].data.slice(start, end + 1)
                                    // Update x-axis ticks with the visible subjects
                                    chart.chart.options.scales.x.ticks.callback = function (value, index, values) {
                                        return points[index] ? extractSubjectFromGifPath(points[index].gif) : '';
                                    };
                                    chart.chart.update();
                                }
                            }
                        },
                        tooltip: {
                            callbacks: {
                                label: function (tooltipItem, data) {
                                    subject = tooltipItem.raw.gif;
                                    return extractSubjectFromGifPath(subject);
                                },
                            }
                        },
                    },
                    onAfterUpdate: function (chart) {
                        console.log('Updating chart');
                        chart.options.scales.x.ticks.callback = function (value, index, values) {
                            console.log(value, index, values);
                            return entropyData.subjects[index];
                        };
                    },
                }
            }
            );
            chartEntropy = chart;
        });
});

