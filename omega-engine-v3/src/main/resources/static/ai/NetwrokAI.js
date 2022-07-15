
/**
 * network + genetic
 */
var AF_AI = {};

AF_AI.network = {};

AF_AI.genetic = {};

AF_AI.utils = {};

AF_AI.network.active = {};

AF_AI.genetic.instance = function(){
	this.populationCount = 0,
	this.chromosomeCount = 0;
	this.crossoverRate = 0.7;
	this.mutationRate = 0.05;
	this.population = null;
	
	this.init = function(){
	}
	
	this.run = function(populationData){
		this.population = new AF_AI.genetic.population(this,this.populationCount,this.chromosomeCount,populationData);
		return this.population.run();
	}
	
	this.calculateFitnessFunction = null;
	
}

AF_AI.genetic.population = function(instance,pCount,cCount,populationData){
	this.count = pCount;
	this.instance = instance;
	this.bestFitness = 0.0;
	this.bestChromosome = null;
	this.bestIndex = 0;
	this.totalFitness = 0.0;
	this.chromosomes = {};
	this.candidate = null;
	this.populationData = populationData;
	
	this.init = function(){
		
		this.candidate = new Array(this.count);
		
		for(let i = 0;i<this.count;i++){
			let chromData = this.populationData[i];
			this.chromosomes[i] = new AF_AI.genetic.chromosome(i,cCount,chromData);
			this.chromosomes[i].fitness = this.instance.calculateFitnessFunction(i);
			
		}
		
	}
	
	this.select = function(){
		
		//compute the probability
		for(let i = 0;i<this.count;i++){

			this.chromosomes[i].probability = this.chromosomes[i].fitness / this.totalFitness;
			
			if(i == 0){
//				console.log(this.chromosomes[i].probability);
				this.chromosomes[i].totalProbability = this.chromosomes[i].probability;
	        }else{
	        	this.chromosomes[i].totalProbability = this.chromosomes[i - 1].totalProbability + this.chromosomes[i].probability;
	        }
		
		}
		
		//select candidate chromosome
		for(let i = 0;i<this.count;i++){
			
			let r = Math.random();
			for(let j = 0;j<this.count;j++){
				
				if(r <= this.chromosomes[j].totalProbability){
//					console.log(this.chromosomes[j].codes);
					this.candidate[i] = this.chromosomes[j].codes;
					break;
				}
			}

		}
		
	}
	
	/**
	 * corss compute
	 */
	this.cross = function(){
		//console.log("candiate1:"+this.candidate[0].length);
		for(let i = 0;i<this.count;i++){
			
			if(Math.random() <= this.instance.crossoverRate){

				let index = AF_AI.utils.randomNotEqInt(this.count,i) - 1;  //get corss index
				
				let positlion = AF_AI.utils.randomInt(cCount);  //get corss positlion
				
				let temp = this.candidate[i][positlion];
				
				this.candidate[i][positlion] = this.candidate[index][positlion]; 
				
				this.candidate[index][positlion] = temp;
				
			}

		}
		
	}
	
	/**
	 * mutation compute
	 */
	this.mutation = function(){
		
		for(let i = 0;i<this.count;i++){
			
			if(Math.random() <= this.instance.mutationRate){
				
				let positlion = AF_AI.utils.randomInt(cCount);  //get mutation positlion
				
				this.candidate[i][positlion] = AF_AI.utils.getNumberInND(0,1);

//				console.log(positlion,this.candidate[i][positlion]);
				
			}

		}
		
	}
	
	this.run = function(){
		
		this.init();
		
		var calculateFitness = AF_AI.genetic.calculateFitness(this.chromosomes);
		
		this.totalFitness = calculateFitness.totalFitness;
		this.bestFitness = calculateFitness.fitness;
		this.bestChromosome = this.chromosomes[calculateFitness.index];
		this.bestIndex = calculateFitness.index;
		
		console.log(calculateFitness);
		
		this.select();
		
		this.cross();
		
		this.mutation();
		
		return this.candidate;
		
	}
	
}

/**
 * codeType:
 * 0: 无编码
 * 1: 二进制编码
 */
AF_AI.genetic.chromosome = function(index,cCount,chromData){
	this.index = index;
	this.count = cCount;
	this.codes = null;
	this.codeType = 0;
	this.fitness = 0.0;
	this.probability = 0.0;
	this.totalProbability = 0.0;
	
	if(this.codeType == 0){
		this.codes = chromData;
	}else{
		this.codes = AF_AI.genetic.decode(chromData);
	}
	
}

AF_AI.genetic.calculateFitness = function(chromosomes){
	var current = 0.0;
	var index = 0;
	var totalFitness = 0.0;
	for(var key in chromosomes){
		
		totalFitness += chromosomes[key].fitness * 1;
		
		if(current < chromosomes[key].fitness * 1){
			current = chromosomes[key].fitness;
			index = key;
		}
	}

	return {totalFitness:totalFitness,index:index,fitness:current};
}

AF_AI.genetic.decode = function(chromData){
	return chromData;
}

AF_AI.network.instance = function(){
	this.layers = new Array();
	this.output = new Array();
	
	this.init = function(){
		for(var i = 0;i<this.layers.length;i++){
			this.layers[i].init();
		}
	}
	
	this.addLayer = function(layer){
		this.layers.push(layer);
	}
	
	this.setPararms = function(params){
		
	}
	
	this.forword = function(input){
		
		var current = input;
		
		for(var i = 0;i<this.layers.length;i++){
			current = this.layers[i].forword(current);
		}
		
		this.output = current;
		
		return this.output;
	}
	
	this.update = function(params){

		var cli = 0;
		//console.log("params",params);
		for(var i = 0;i<this.layers.length;i++){
			
			/**
			 * 跳过输入层
			 */
			if(this.layers[i].layerType != 0){

				var weight = AF_AI.utils.zero2d(this.layers[i].inputNum,this.layers[i].outputNum);
				
				var bias = AF_AI.utils.zero(this.layers[i].outputNum);
				
				var startWIndex = cli * this.layers[i].inputNum * this.layers[i].outputNum;
				
				var startBIndex = cli * this.layers[i].outputNum;
				
				for(var x = 0;x<this.layers[i].inputNum;x++){
					for(var y = 0;y<this.layers[i].outputNum;y++){
						weight[x][y] = params.w[startWIndex + x * this.layers[i].outputNum + y];
					}
				}
				
				for(var x = 0;x<this.layers[i].outputNum;x++){
					bias[x] = params.b[startBIndex + x];
				}
				//console.log(bias);
				this.layers[i].update(weight,bias);
				
				cli++;
			}
			
		}
		
	}
	
	this.updateV2 = function(params){

		var index = 0;
		
		//console.log("params",params);
		for(var i = 0;i<this.layers.length;i++){
			
			/**
			 * 跳过输入层
			 */
			if(this.layers[i].layerType != 0){

				var weight = AF_AI.utils.zero2d(this.layers[i].inputNum,this.layers[i].outputNum);
				
				var bias = AF_AI.utils.zero(this.layers[i].outputNum);

				//console.log("start——index",index);
				
				for(var x = 0;x<this.layers[i].inputNum;x++){
					for(var y = 0;y<this.layers[i].outputNum;y++){
						
						weight[x][y] = params.data[index];
						index++;
					}
				}
				
				for(var x = 0;x<this.layers[i].outputNum;x++){
					
					bias[x] = params.data[index];
					index++;
				}
				//console.log(bias);
				this.layers[i].update(weight,bias);
				
				//console.log("end——index",index);
			}
			
		}
		
	}
	
	this.setParams = function(params){
		
		var index = 0;
		//console.log(params.length);
		for(var i = 0;i<this.layers.length;i++){
			
			/**
			 * 跳过输入层
			 */
			if(this.layers[i].layerType != 0){
				var weight = AF_AI.utils.zero2d(this.layers[i].inputNum,this.layers[i].outputNum);
				var bias = AF_AI.utils.zero(this.layers[i].outputNum);

				for(var x = 0;x<this.layers[i].inputNum;x++){
					for(var y = 0;y<this.layers[i].outputNum;y++){
						weight[x][y] = params[index];
						index++;
					}
				}
				
				for(var x = 0;x<this.layers[i].outputNum;x++){
					bias[x] = params[index];
					index++;
				}
				//console.log(bias);
				this.layers[i].update(weight,bias);
				//console.log("end——index",index);
			}
			
		}
		
	}
	
	this.getParams = function(){
		var params = {w:new Array(),b:new Array()};
		
		for(var i = 0;i<this.layers.length;i++){
			
			if(this.layers[i].layerType != 0){

				var currentW = AF_AI.utils.matrix2dToArray(this.layers[i].weight);
				var currentB = this.layers[i].bias;
				
				for(var j = 0;j<currentW.length;j++){
					params.w.push(currentW[j]);
				}
				for(var j = 0;j<currentB.length;j++){
					params.b.push(currentB[j]);
				}
				//console.log(params.b);
			}
			
		}
		
		return params;
	}
	
	this.getParamsV2 = function(){
		var params = new Array();
		
		for(var i = 0;i<this.layers.length;i++){
			
			if(this.layers[i].layerType != 0){

				var currentW = AF_AI.utils.matrix2dToArray(this.layers[i].weight);
				var currentB = this.layers[i].bias;
				
				for(var j = 0;j<currentW.length;j++){
					params.push(currentW[j]);
				}
				for(var j = 0;j<currentB.length;j++){
					params.push(currentB[j]);
				}
			}
			
		}
		
		return params;
	}
	
}

/**
 * layerType:
 * 0: inputLayer
 * 1: fullyLayer
 * activeType:
 * 0: sigmoid
 * 1: relu
 */
AF_AI.network.layer = function(){
	this.layerType = 0;
	this.name = null;
	this.inputNum = 0;
	this.outputNum = 0;
	this.activeType = null;
	this.weight = null;
	this.bias = null;
	this.input = null;
	this.output = null;
	
	this.init = function(){
		if(this.layerType != 0){
			this.weight = AF_AI.utils.matrix2d(this.inputNum,this.outputNum);
			this.bias = AF_AI.utils.matrix(this.outputNum);
		}
	}
	
	this.forword = function(input){

		this.input = input;
		
		if(this.layerType == 0){
			this.output = this.input;
		}else{
	
			this.output = AF_AI.utils.zero(this.outputNum);
			
			for(var o = 0;o<this.outputNum;o++){
				for(var i = 0;i<this.inputNum;i++){
					
					this.output[o] = this.output[o] * 1 + this.input[i] * this.weight[i][o];
				}
				this.output[o] = this.output[o] + this.bias[o] * 1;
			}
			
			switch (this.activeType) {
			case 0:
				this.output = AF_AI.network.active.sigmoid(this.output);
				break;
			case 1:
				this.output = AF_AI.network.active.relu(this.output);
				break;
			default:
				break;
			}
			
		}
		
		return this.output;
	}
	
	this.update = function(weight,bias){
		this.weight = weight;
		this.bias = bias;
	}
	
}

AF_AI.network.active.sigmoid = function(x){
	var y = new Array(x.length);
	for(var i = 0;i<x.length;i++) {
		y[i] = (1 / (1 + Math.exp(-x[i])));
	}
	return y;
}

AF_AI.network.active.relu = function(x){
	var y = new Array(x.length);
	for(var i = 0;i<x.length;i++) {
		if(x[i] > 0) {
			y[i] = x[i];
		}else {
			y[i] = 0;
		}
	}
	return y;
}

AF_AI.utils.zero = function(n){
	var x = new Array(n);
	for(var i = 0;i<n;i++){
		x[i] = 0;
	}
	return x;
}

AF_AI.utils.zero2d = function(n,m){
	var x = new Array(n);
	for(var i = 0;i<n;i++){
		var y = new Array(m);
		for(var j = 0;j<m;j++){
			y[j] = 0;
		}
		x[i] = y;
	}
	return x;
}

AF_AI.utils.matrix = function(n){
	var x = new Array(n);
	for(var i = 0;i<n;i++){
		x[i] = AF_AI.utils.getNumberInND();
	}
	return x;
}

AF_AI.utils.matrix2d = function(n,m){
	var x = new Array(n);
	for(var i = 0;i<n;i++){
		var y = new Array(m);
		for(var j = 0;j<m;j++){
			y[j] = AF_AI.utils.getNumberInND();
		}
		x[i] = y;
	}
	return x;
}

AF_AI.utils.getNumberInND = function(mean,std_dev){
	if(mean = null || std_dev == null){
		mean = 0;
		std_dev = 1;
	}
    return mean+(AF_AI.utils.randomND()*std_dev);
}

AF_AI.utils.getNumbersInND = function(mean,std_dev,count){
	var vals = new Array();
	if(mean = null || std_dev == null){
		mean = 0;
		std_dev = 1;
	}
	
	for(let i = 0;i<count;i++){
		vals.push(mean+(AF_AI.utils.randomND()*std_dev));
	}
	
    return vals;
}

AF_AI.utils.randomND = function(){
    var u=0.0, v=0.0, w=0.0, c=0.0;
    do{
        //获得两个（-1,1）的独立随机变量
        u=Math.random()*2-1.0;
        v=Math.random()*2-1.0;
        w=u*u+v*v;
    }while(w==0.0||w>=1.0)
    //这里就是 Box-Muller转换
    c=Math.sqrt((-2*Math.log(w))/w);
    //返回2个标准正态分布的随机数，封装进一个数组返回
    //当然，因为这个函数运行较快，也可以扔掉一个
    //return [u*c,v*c];
    return u*c;
}

AF_AI.utils.randomInt = function(x){
    return Math.ceil(Math.random() * x);
}

AF_AI.utils.randomNotEqInt = function(x,y){
	var temp = Math.ceil(Math.random() * x);
	if(temp == y){
		temp = AF_AI.utils.randomNotEqInt(x,y);
	}
    return temp;
}

AF_AI.utils.matrix2dToArray = function(x){
	var y = new Array();
	for(var i = 0;i<x.length;i++){
		for(var j = 0;j<x[i].length;j++){
			y[i * x[i].length + j * 1] = x[i][j];
		}
	}
    return y;
}
