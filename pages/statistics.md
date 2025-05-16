---
layout: single
classes: wide
title: Statistics
permalink: /statistics
section: statistics
---

Statistics about subject

<div id="statsContainer">
        <select id="statsDropdown" onchange="updateStatsPlot(this.value)">
            <!-- GIF options will be populated here -->
        </select>
        <br>
        <div id="radioContainer">
            <input type="radio" id="meanRadio" name="statsOption" value="mean" checked>
            <label for="meanRadio">Mean</label>
            <input type="radio" id="stdRadio" name="statsOption" value="std">
            <label for="stdRadio">Std</label>
            <input type="radio" id="sigRadio" name="statsOption" value="sig">
            <label for="sigRadio">Sig</label>
        </div>
</div>
<br>
<div id="thicknessContainer">
    <p>Cortical thickness</p>
    <div id="thicknessChart"></div>
</div>
<br>
<div id="areaContainer">
    <p> Surface area </p>
    <div id="areaChart"></div>
</div>
<br>
<div id="volumeContainer">
    <p>Subcortical volume</p>
    <div id="volumeChart"></div>
</div>

<style>
    #statsContainer {
        display: flex;
        align-items: center;

    }
    #radioContainer input[type="radio"] {
        display: inline-block;
        margin-right: 5px;
        margin-left: 20px;
    }
    #radioContainer label {
        display: inline-block;
    }
</style>
