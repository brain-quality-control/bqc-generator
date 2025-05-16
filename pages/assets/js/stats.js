const excludedROIthickness = ["subjects", "hemi", "BrainSegVolNotVent", "eTIV"];
const excludedROIarea = ["subjects", "hemi", "BrainSegVolNotVent", "eTIV", "WhiteSurfArea_area"];
const excludedROIvolume = ["subjects", "BrainSegVolNotVent", "BrainSegVol", "CortexVol", "TotalGrayVol", "SupraTentorialVol", "EstimatedTotalIntraCranialVol", "MaskVol", "lhCortexVol", "rhCortexVol", "lhCerebralWhiteMatterVol", "rhCerebralWhiteMatterVol", "CerebralWhiteMatterVol", "SupraTentorialVolNotVent", "SubCortGrayVol"];

const excludedROI = {
	"volume": excludedROIvolume,
	"thickness": excludedROIthickness,
	"area": excludedROIarea
}

var selectedStat;
var currentSubjectIndex = 0;

function clipArray(array, minVal, maxVal) {
	return array.map(val => Math.max(minVal, Math.min(maxVal, val)));
}

function onStatisticsPage() {
	return document.URL.includes('statistics');
}

function extractSubjectFromGifPath(gifPath) {
	return gifPath.split('/').pop().split('.').slice(0, -1).join('.');
}


function capitalizeFirstLetter(string) {
	return string.charAt(0).toUpperCase() + string.slice(1);
}

function _updateStatsPlot(gifPath, type) {
	const selectedSubject = extractSubjectFromGifPath(gifPath);
	selectedStat = document.querySelector('input[name="statsOption"]:checked').value;
	console.log("updateStatsPlot called", gifPath, type, selectedSubject, selectedStat);


	drawStatistic(selectedSubject, 'volume', selectedStat, 'volume', 'Subcortical Volume Mean', type);
	drawStatistic(selectedSubject, 'thickness', selectedStat, 'thickness', 'Cortical thickness Mean', type);
	drawStatistic(selectedSubject, 'area', selectedStat, 'surface area', 'Surface Area', type);
}

function updateStatsPlot(subjectIndex) {
	console.log("updateStatsPlot called", subjectIndex);
	console.log(document);
	if (onStatisticsPage()) {
		currentSubjectIndex = subjectIndex;
		subjectPath = jsonData[subjectIndex];
		_updateStatsPlot(subjectPath.gif, 'scatter');
	}
}

function fillStatsDropdown() {
	const gifDropdown = document.getElementById('statsDropdown');
	jsonData.forEach((item, index) => {
		const basename = item["gif"].split('/').pop();
		const option = document.createElement('option');
		option.value = index;
		option.text = extractSubjectFromGifPath(item["gif"]);
		// option.text = basename.split('.').slice(0, -1).join('.');
		gifDropdown.appendChild(option);
	});
	if (jsonData.length > 0) {
		gifDropdown.selectedIndex = 0;
	}
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
			if (onStatisticsPage()) {
				fillStatsDropdown();
				_updateStatsPlot(jsonData[0].gif);
			}
		});

	// Add this code after your radio buttons in your HTML
	document.querySelectorAll('input[name="statsOption"]').forEach((radio) => {
		console.log('Adding event listener to radio button');
		radio.addEventListener('change', () => {
			updateStatsPlot(currentSubjectIndex);
		});
	});

});

function drawStatisticsNoHemisphere(selectedData, measure, name, title, type) {
	let x = [], y = [];

	console.log(excludedROI[measure].values());
	console.log(excludedROI[measure].includes("subjects"));
	for (let i = 0; i < selectedData.length; i++) {
		let keys = Object.keys(selectedData[i]).filter(key => !excludedROI[measure].includes(key));
		let values = keys.map(key => selectedData[i][key]);

		x.push(...keys);
		y.push(...values);
	}

	if (selectedStat === 'sig') {
		y = clipArray(y, 0, 7);
	}

	const trace = {
		x: x,
		y: y,
		type: type,
		name: name,
		mode: 'markers'
	};

	const layout = {
		title: title
	};

	return Plotly.newPlot(`${measure}Chart`, [trace], layout);
}

function drawStatisticsHemisphere(selectedData, measure, name, title, type) {
	var traces = []
	for (const hemi of ["lh", "rh"]) {
		const hemiData = selectedData.filter(item => item.hemi === hemi);
		let x = [], y = [];
		console.log(hemiData);
		for (let i = 0; i < hemiData.length; i++) {
			let keys = Object.keys(hemiData[i]).filter(key => !excludedROI[measure].includes(key));
			console.log(keys);
			let values = keys.map(key => hemiData[i][key]);

			x.push(...keys);
			y.push(...values);
		}

		if (selectedStat === 'sig') {
			y = clipArray(y, 0, 7);
		}


		const trace = {
			x: x,
			y: y,
			type: type,
			name: hemi,
			mode: 'markers'
		};

		traces.push(trace);
	}
	const layout = {
		title: title
	};
	return Plotly.newPlot(`${measure.toLowerCase()}Chart`, traces, layout);
}

function drawStatistic(subject, measure, stat, name, title, type) {
	const csvFilePath = `/static/stats/${measure}_${stat}.csv`;

	Papa.parse(csvFilePath, {
		download: true,
		header: true,
		complete: function (results) {
			const data = results.data;
			const selectedData = data.filter(item => item.subjects === subject);

			if (measure === 'volume') {
				return drawStatisticsNoHemisphere(selectedData, measure, name, title, type);
			} else {
				return drawStatisticsHemisphere(selectedData, measure, name, title, type);
			}
		}
	});
}

