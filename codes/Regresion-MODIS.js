/**** Start of imports. If edited, may not auto-convert in the playground. ****/
var geometry = ee.FeatureCollection("projects/umutbediroglu/assets/states"),
    visl8 = {"min":250,"max":350,"palette":["#000080","#0000D9","#4000FF","#8000FF","#0080FF","#00FFFF","#00FF80","#80FF00","#DAFF00","#FFFF00","#FFF500","#FFDA00","#FFB000","#FFA400","#FF4F00","#FF2500","#FF0A00","#FF00FF"]},
    TemperatureVis = {"min":250,"max":350,"palette":["040274","040281","0502a3","0502b8","0502ce","0502e6","0602ff","235cb1","307ef3","269db1","30c8e2","32d3ef","3be285","3ff38f","86e26f","3ae237","b5e22e","d6e21f","fff705","ffd611","ffb613","ff8b13","ff6e08","ff500d","ff0000","de0101","c21301","a71001","911003"]};
/***** End of imports. If edited, may not auto-convert in the playground. *****/
var monthly_temp = ee.FeatureCollection("projects/umutbediroglu/assets/station_monthly_temp")
var modisdata = ee.ImageCollection('MODIS/006/MOD11A1')

function modisfunc(image){
  var modis = image.select('LST_Day_.*').multiply(0.02);
  return image.addBands(modis,null,true)
}

var data_modis = modisdata.map(modisfunc);
var modisLST = data_modis.select('LST_Day_1km')
.filter(ee.Filter.date('2021-03-01', '2021-03-31'))
.mean().clip(geometry);

Map.addLayer(modisLST, TemperatureVis, 'Air Temperature MODIS');
Map.addLayer(monthly_temp, {}, "Station Air Temperature");




var tempBands = modisLST;
var label = "TEMP";
var bands = ["LST_Day_1km"];
var input = tempBands.select(bands);
var training = monthly_temp;

var trainingImage = tempBands.sampleRegions({
  collection: training,
  properties: [label],
  scale: 30
});

var trainingData = trainingImage.randomColumn();
var trainSet = trainingData.filter(ee.Filter.lt ("random", 0.8));
var validation = trainingData.filter(ee.Filter.lt ("random", 0.2));

var RFR = ee.Classifier.smileRandomForest(1500).setOutputMode("REGRESSION")
  .train(trainSet.select(["LST_Day_1km", "TEMP"]),
  'TEMP', ["LST_Day_1km"])

var arr = trainSet.classify(RFR, "Estimation")
var aar_v = validation.classify(RFR, "Estimation Validation")

print(ui.Chart.feature.byFeature({
  features: arr,
  xProperty: "TEMP",
  yProperties: ["Estimation"]})
  .setChartType('ScatterChart')
  .setOptions({
    legend: {position: 'none'},
    hAxis: {'title': 'Station Air Temperature'},
    vAxis: {'title': 'Estimation RFR Train'},
    series: {
      0: {
        pointSize: 3,
        dataOpacity: 0.5,
      },
      1: {
        pointSize: 5,
        lineWidth: 2,
      }
    }
  })
);

print(ui.Chart.feature.byFeature({
  features: aar_v,
  xProperty: "TEMP",
  yProperties: ["Estimation Validation"]})
  .setChartType('ScatterChart')
  .setOptions({
    legend: {position: 'none'},
    hAxis: {'title': 'Station Air Temperature'},
    vAxis: {'title': 'Estimation RFR Validation'},
    series: {
      0: {
        pointSize: 3,
        dataOpacity: 0.5,
      },
      1: {
        pointSize: 5,
        lineWidth: 2,
      }
    }
  })
);
var classified1 = input.classify(RFR)
Map.addLayer(classified1,TemperatureVis,'Classification RFR')

var correl = ee.Reducer.pearsonsCorrelation();
var reduced = arr.reduceColumns(correl, ['TEMP', 'Estimation'])
print('Pearsons Correlation Coefficient (r) Train: ',reduced)

var correl_1 = ee.Reducer.pearsonsCorrelation();
var reduced_1 = aar_v.reduceColumns(correl_1, ['TEMP', 'Estimation Validation'])
print('Pearsons Correlation Coefficient (r) Validation: ',reduced_1)

var sheds = ee.FeatureCollection(arr)
var areaDiff = function(feature) {
  var area = ee.Number(feature.get('TEMP'));
  var diff = area.subtract(ee.Number(feature.get('Estimation')));
  return feature.set('diff', diff.pow(2));
};

var rmse = ee.Number(
  sheds.map(areaDiff)
  .reduceColumns(ee.Reducer.mean(), ['diff'])
  .get('mean')
)
.sqrt();
print('RMSE Train: ', rmse);

var sheds = ee.FeatureCollection(aar_v)
var areaDiff = function(feature) {
  var area = ee.Number(feature.get('TEMP'));
  var diff = area.subtract(ee.Number(feature.get('Estimation Validation')));
  return feature.set('diff', diff.pow(2));
};

var rmse = ee.Number(
  sheds.map(areaDiff)
  .reduceColumns(ee.Reducer.mean(), ['diff'])
  .get('mean')
)
.sqrt();
print('RMSE Validation: ', rmse);




var tempBands = modisLST;
var label = "TEMP";
var bands = ["LST_Day_1km"];
var input = tempBands.select(bands);
var training = monthly_temp;

var trainingImage = tempBands.sampleRegions({
  collection: training,
  properties: [label],
  scale: 30
});

var trainingData = trainingImage.randomColumn();
var trainSet = trainingData.filter(ee.Filter.lt ("random", 0.8));
var validation = trainingData.filter(ee.Filter.lt ("random", 0.2));

var SVR1 = ee.Classifier.libsvm({svmType:"EPSILON_SVR", kernelType: "RBF", gamma:0.01, cost:2048}).setOutputMode("REGRESSION")
  .train(trainSet.select(["LST_Day_1km", "TEMP"]),
  'TEMP', ["LST_Day_1km"])
  
var arr = trainSet.classify(SVR1, "Estimation")
var aar_v = validation.classify(SVR1, "Estimation Validation")

print(ui.Chart.feature.byFeature({
  features: arr,
  xProperty: "TEMP",
  yProperties: ["Estimation"]})
  .setChartType('ScatterChart')
  .setOptions({
    legend: {position: 'none'},
    hAxis: {'title': 'Station Air Temperature'},
    vAxis: {'title': 'Estimation SVR-RBF Train'},
    series: {
      0: {
        pointSize: 3,
        dataOpacity: 0.5,
      },
      1: {
        pointSize: 5,
        lineWidth: 2,
      }
    }
  })
);

print(ui.Chart.feature.byFeature({
  features: aar_v,
  xProperty: "TEMP",
  yProperties: ["Estimation Validation"]})
  .setChartType('ScatterChart')
  .setOptions({
    legend: {position: 'none'},
    hAxis: {'title': 'Station Air Temperature'},
    vAxis: {'title': 'Estimation SVR-RBF Validation'},
    series: {
      0: {
        pointSize: 3,
        dataOpacity: 0.5,
      },
      1: {
        pointSize: 5,
        lineWidth: 2,
      }
    }
  })
);
var classified2 = input.classify(SVR1)
Map.addLayer(classified2,TemperatureVis,'Classification RBF SVR')

var correl = ee.Reducer.pearsonsCorrelation();
var reduced = arr.reduceColumns(correl, ['TEMP', 'Estimation'])
print('Pearsons Correlation Coefficient (r) Train: ',reduced)

var correl_1 = ee.Reducer.pearsonsCorrelation();
var reduced_1 = aar_v.reduceColumns(correl_1, ['TEMP', 'Estimation Validation'])
print('Pearsons Correlation Coefficient (r) Validation: ',reduced_1)

var sheds = ee.FeatureCollection(arr)
var areaDiff = function(feature) {
  var area = ee.Number(feature.get('TEMP'));
  var diff = area.subtract(ee.Number(feature.get('Estimation')));
  return feature.set('diff', diff.pow(2));
};

var rmse = ee.Number(
  sheds.map(areaDiff)
  .reduceColumns(ee.Reducer.mean(), ['diff'])
  .get('mean')
)
.sqrt();
print('RMSE Train: ', rmse);

var sheds = ee.FeatureCollection(aar_v)
var areaDiff = function(feature) {
  var area = ee.Number(feature.get('TEMP'));
  var diff = area.subtract(ee.Number(feature.get('Estimation Validation')));
  return feature.set('diff', diff.pow(2));
};

var rmse = ee.Number(
  sheds.map(areaDiff)
  .reduceColumns(ee.Reducer.mean(), ['diff'])
  .get('mean')
)
.sqrt();
print('RMSE Validation: ', rmse);




var tempBands = modisLST;
var label = "TEMP";
var bands = ["LST_Day_1km"];
var input = tempBands.select(bands);
var training = monthly_temp;

var trainingImage = tempBands.sampleRegions({
  collection: training,
  properties: [label],
  scale: 30
});

var trainingData = trainingImage.randomColumn();
var trainSet = trainingData.filter(ee.Filter.lt ("random", 0.8));
var validation = trainingData.filter(ee.Filter.lt ("random", 0.2));

var SVR2 = ee.Classifier.libsvm({svmType:"EPSILON_SVR", kernelType: "LINEAR", cost:50}).setOutputMode("REGRESSION")
  .train(trainSet.select(["LST_Day_1km", "TEMP"]),
  'TEMP', ["LST_Day_1km"])

var arr = trainSet.classify(SVR2, "Estimation")
var aar_v = validation.classify(SVR2, "Estimation Validation")


print(ui.Chart.feature.byFeature({
  features: arr,
  xProperty: "TEMP",
  yProperties: ["Estimation"]})
  .setChartType('ScatterChart')
  .setOptions({
    legend: {position: 'none'},
    hAxis: {'title': 'Station Air Temperature'},
    vAxis: {'title': 'Estimation SVR-LINEAR Train'},
    series: {
      0: {
        pointSize: 3,
        dataOpacity: 0.5,
      },
      1: {
        pointSize: 5,
        lineWidth: 2,
      }
    }
  })
);

print(ui.Chart.feature.byFeature({
  features: aar_v,
  xProperty: "TEMP",
  yProperties: ["Estimation Validation"]})
  .setChartType('ScatterChart')
  .setOptions({
    legend: {position: 'none'},
    hAxis: {'title': 'Station Air Temperature'},
    vAxis: {'title': 'Estimation SVR-LINEAR Validation'},
    series: {
      0: {
        pointSize: 3,
        dataOpacity: 0.5,
      },
      1: {
        pointSize: 5,
        lineWidth: 2,
      }
    }
  })
);
var classified3 = input.classify(SVR2)
Map.addLayer(classified3,TemperatureVis,'Classification LINEAR SVR')

var correl = ee.Reducer.pearsonsCorrelation();
var reduced = arr.reduceColumns(correl, ['TEMP', 'Estimation'])
print('Pearsons Correlation Coefficient (r) Train: ',reduced)

var correl_1 = ee.Reducer.pearsonsCorrelation();
var reduced_1 = aar_v.reduceColumns(correl_1, ['TEMP', 'Estimation Validation'])
print('Pearsons Correlation Coefficient (r) Validation: ',reduced_1)

var sheds = ee.FeatureCollection(arr)
var areaDiff = function(feature) {
  var area = ee.Number(feature.get('TEMP'));
  var diff = area.subtract(ee.Number(feature.get('Estimation')));
  return feature.set('diff', diff.pow(2));
};

var rmse = ee.Number(
  sheds.map(areaDiff)
  .reduceColumns(ee.Reducer.mean(), ['diff'])
  .get('mean')
)
.sqrt();
print('RMSE Train: ', rmse);

var sheds = ee.FeatureCollection(aar_v)
var areaDiff = function(feature) {
  var area = ee.Number(feature.get('TEMP'));
  var diff = area.subtract(ee.Number(feature.get('Estimation Validation')));
  return feature.set('diff', diff.pow(2));
};

var rmse = ee.Number(
  sheds.map(areaDiff)
  .reduceColumns(ee.Reducer.mean(), ['diff'])
  .get('mean')
)
.sqrt();
print('RMSE Validation: ', rmse);




var trainingImage = tempBands.sampleRegions({
  collection: training,
  properties: [label],
  scale: 30
});

var trainingData = trainingImage.randomColumn();
var trainSet = trainingData.filter(ee.Filter.lt ("random", 0.8));
var validation = trainingData.filter(ee.Filter.lt ("random", 0.2));

var smileCart = ee.Classifier.smileCart(1500).setOutputMode("REGRESSION")
.train(trainSet, label, bands);
var arr = trainSet.classify(smileCart, "Estimation")
var aar_v = validation.classify(smileCart, "Estimation Validation")

print(ui.Chart.feature.byFeature({
  features: arr,
  xProperty: "TEMP",
  yProperties: ["Estimation"]})
  .setChartType('ScatterChart')
  .setOptions({
    legend: {position: 'none'},
    hAxis: {'title': 'Station Air Temperature'},
    vAxis: {'title': 'Estimation smileCart Train'},
    series: {
      0: {
        pointSize: 3,
        dataOpacity: 0.5,
      },
      1: {
        pointSize: 5,
        lineWidth: 2,
      }
    }
  })
);

print(ui.Chart.feature.byFeature({
  features: aar_v,
  xProperty: "TEMP",
  yProperties: ["Estimation Validation"]})
  .setChartType('ScatterChart')
  .setOptions({
    legend: {position: 'none'},
    hAxis: {'title': 'Station Air Temperature'},
    vAxis: {'title': 'Estimation smileCart Validation'},
    series: {
      0: {
        pointSize: 3,
        dataOpacity: 0.5,
      },
      1: {
        pointSize: 5,
        lineWidth: 2,
      }
    }
  })
);
var classified4 = input.classify(smileCart)
Map.addLayer(classified4,TemperatureVis,'Classification smileCart')
Map.centerObject(geometry);

var correl = ee.Reducer.pearsonsCorrelation();
var reduced = arr.reduceColumns(correl, ['TEMP', 'Estimation'])
print('Pearsons Correlation Coefficient (r) Train: ',reduced)

var correl_1 = ee.Reducer.pearsonsCorrelation();
var reduced_1 = aar_v.reduceColumns(correl_1, ['TEMP', 'Estimation Validation'])
print('Pearsons Correlation Coefficient (r) Validation: ',reduced_1)

var sheds = ee.FeatureCollection(arr)
var areaDiff = function(feature) {
  var area = ee.Number(feature.get('TEMP'));
  var diff = area.subtract(ee.Number(feature.get('Estimation')));
  return feature.set('diff', diff.pow(2));
};

var rmse = ee.Number(
  sheds.map(areaDiff)
  .reduceColumns(ee.Reducer.mean(), ['diff'])
  .get('mean')
)
.sqrt();
print('RMSE Train: ', rmse);

var sheds = ee.FeatureCollection(aar_v)
var areaDiff = function(feature) {
  var area = ee.Number(feature.get('TEMP'));
  var diff = area.subtract(ee.Number(feature.get('Estimation Validation')));
  return feature.set('diff', diff.pow(2));
};

var rmse = ee.Number(
  sheds.map(areaDiff)
  .reduceColumns(ee.Reducer.mean(), ['diff'])
  .get('mean')
)
.sqrt();
print('RMSE Validation: ', rmse);




function makeColorBarParams(palette) {
  return {
    bbox: [0, 0, 1, 0.1],
    dimensions: '100x10',
    format: 'png',
    min: 0,
    max: 1,
    palette: palette,
  };
}

var colorBar = ui.Thumbnail({
  image: ee.Image.pixelLonLat().select(0),
  params: makeColorBarParams(TemperatureVis.palette),
  style: {stretch: 'horizontal', margin: '0px 8px', maxHeight: '24px'},
});

var legendLabels = ui.Panel({
  widgets: [
    ui.Label(TemperatureVis.min, {margin: '4px 8px'}),
    ui.Label(
        ((TemperatureVis.max-TemperatureVis.min) / 2+TemperatureVis.min),
        {margin: '4px 8px', textAlign: 'center', stretch: 'horizontal'}),
    ui.Label(TemperatureVis.max, {margin: '4px 8px'})
  ],
  layout: ui.Panel.Layout.flow('horizontal')
});

var legendTitle = ui.Label({
  value: 'March 2021 Mean Air Temperature (K)',
  style: {fontWeight: 'bold'}
});

var legendPanel = ui.Panel([legendTitle, colorBar, legendLabels]);
Map.add(legendPanel);
