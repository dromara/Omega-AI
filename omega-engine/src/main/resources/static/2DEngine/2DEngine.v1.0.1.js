/**
 * 2d Engine Experiment
 * 
 * Copyright (c) 2018 leo
 * 
 * @author leo
 * @date 2020-08-11
 * @version 1.0.1
 * @email 465973119@qq.com
 * @remark 
 * 		version 1.0.1
 * 		update in 2020-08-11 18:40:10
 * 		update remark:
 * 			1.添加click,mousedown,mousemove,mouseup事件监听
 * 			2.优化坐标整形计算提高fps
 * 			3.优化播放动画方法
 * 			4.优化原始图形统一渲染
 * 			5.添加离屏渲染
 *          6.添加线与方形碰撞测试
 *      version 1.0.1.1
 * 		update in 2020-12-07 21:40:10
 * 		update remark:
 *          1.添加方形与方形碰撞测试
 *          2.更换线段与线段碰撞测试
 */
var Engine2D = {};

Engine2D.fps = 0;

Engine2D.lastCalledTime = 0;

Engine2D.method = {};

Engine2D.label = {};

Engine2D.play = function(fn,o){
//	console.log("in");
	fn(o);
	requestAnimationFrame(function(){
		Engine2D.play(fn,o);
	});
}

Engine2D.instance = function(){
	this.name = null;
	this.canvas = null;
	this.cacheCanvas = null;
	this.context = null;
	this.mouseX = 0;
	this.mouseY = 0;
	this.width = 0;
	this.height = 0;
	this.fps = 30;
	this.showFPS = false;
	this.cache = false;
	this.cacheContext = null;
	this.sceneList = {};
	
	this.init = function(canvas){
		if(canvas!=null){
			this.canvas = canvas
			this.context = canvas.getContext('2d');
			this.width = canvas.width;
			this.height = canvas.height;
			if(this.cache){
				this.cacheCanvas = document.createElement("canvas");
				this.cacheCanvas.width = this.width;
				this.cacheCanvas.height = this.height;
				this.cacheContext = this.cacheCanvas.getContext('2d');
			}
			Engine2D.engine.mouseListen(this);
		}
	}
	
	this.addScene = function(scene){
		if(scene != null){
			scene.instance = this;
			scene.context = this.context;
			this.sceneList[scene.name] = scene;
		}
	}
	
	this.runScene = function(sceneName){
		this.sceneList[sceneName].run(this);
	}
	
	this.clear = function(){
		if(this.cache){
			this.cacheContext.clearRect(0,0,this.width,this.height);
		}
		this.context.clearRect(0,0,this.width,this.height);
	}
	
	this.FPS = function(){
		if(this.showFPS){
			if(!Engine2D.lastCalledTime) {
				Engine2D.lastCalledTime = Date.now();
				Engine2D.fps = 0;
			}else{
				var delta = (Date.now() - Engine2D.lastCalledTime)/1000;
			    Engine2D.lastCalledTime = Date.now();
			    Engine2D.fps = 1/delta;
			    console.log(Engine2D.fps);
			}
		}
	}
	
	this.click = function(){
		
	}
	
	this.mouseup = function(){
		
	}
	
	this.mousedown = function(){
		
	}
	
	this.mousemove = function(){
		
	}
	
}

Engine2D.method.getInstance = function(){
	return new Engine2D.instance();
}

Engine2D.scene = function(){
	this.instance = null
	this.name = null;
	this.backgroud = null;
	this.layerList = {};
	this.context = null;
	this.fps = 30;
	this.hitTest = false;
	this.childs = new Array();
	this.rigidBodyChilds = new Array();
	
	this.addLayer = function(layer){
		if(layer != null){
			layer.context = this.context;
			layer.cacheContext = this.instance.cacheContext;
			layer.parent = this;
			layer.scene = this;
			this.layerList[layer.name] = layer;
		}
	}
	
	this.removeChilds = function(name){
		
		for(var i = 0;this.childs.length;i++){
			if(name == this.childs[i].name){
				
				this.childs.remove(i);
				
				break;
			}
		}
		
		for(var i = 0;this.rigidBodyChilds.length;i++){
			if(name == this.rigidBodyChilds[i].name){
				
				this.rigidBodyChilds.remove(i);
				
				break;
			}
		}
		
		for(var key in this.layerList){
			
			for(var key2 in this.layerList[key].spiritList){
				
				if(name == this.layerList[key].spiritList[key2].name){
					delete this.layerList[key].spiritList[key2];
					break;
				}
				
			}
			
		}
		
	}
	
	this.run = function(){

		if(window.requestAnimationFrame){
			Engine2D.play(this.play,this);
		}else{
			throw new Error("not support this browser,plase use the chrome browse open this html.");
//			setInterval(function(){
//				that.play(that);
//			},1000/that.fps);
		}
		
	}
	
	this.play = function(o){
		o.update();
//		o.instance.FPS();
		if(o.hitTest){
			Engine2D.engine.hitTest(o);
			Engine2D.engine.afterHitTest(o);
		}
		for(var key in o.layerList){
			o.layerList[key].run();
		}
	}
	
	this.update = function(){
		
	}
	
	this.afterHitTest = function(){
		
	}
	
}

Engine2D.layer = function(){
	this.name = null;
	this.backgroud = null;
	this.spiritList = {};
	this.context = null;
	this.cacheContext = null;
	this.parent = null;
	this.scene = null;
	
	this.addSpirit = function(spirit){
		if(spirit != null){
			if(this.scene.instance.cache){
				spirit.context = this.cacheContext;
			}else{
				spirit.context = this.context;
			}
			spirit.scene = this.scene;
			spirit.layer = this;
			this.scene.childs.push(spirit);
			if(spirit.substanceType==1){
				this.scene.rigidBodyChilds.push(spirit);
			}
			this.spiritList[spirit.name] = spirit;
		}
	}
	
	this.run = function(){
		this.update();
		this.draw();
	}
	
	this.update = function(){
		
	}
	
	this.draw = function(){
		for(var key in this.spiritList){
			this.spiritList[key].run();
		}
		if(this.scene.instance.cache){
			this.context.drawImage(this.scene.instance.cacheCanvas, 0, 0);
		}
	}
	
}

Engine2D.spirit = function(){
	this.name = null;
	/*
	 * spiritType 0:org,1:image,2:frameAnimation
	 */
	this.spiritType = 0;
	/*
	 * orgType 0:fillRect,1:circles,2:line,3:text
	 */
	this.orgType = 0;
	/*
	 * substanceType 0: fictitious,1: rigidBody
	 */
	this.substanceType = 0;
	this.showName = false;
	this.width = 0;
	this.height = 0;
	this.scaleX = 1;
	this.scaleY = 1;
	this.x = 0;  //fixed x
	this.y = 0;  //fixed y
	this.relativeX = 0;  //relative parent x
	this.relativeY = 0;  //relative parent y
	this.hitAreaWidth = this.width;  //hit area width
	this.hitAreaHeight = this.height;  //hit area height
	this.direction = 0;
	this.rotate = 0;
	this.relativeRotate = 0;  //relative rotate
	/*
	 * position 0:absolute,1:fixed
	 */
	this.position = 0;
	/*
	 * isFollowParent 0:yes,1:no
	 */
	this.isFollowParent = 0;
	this.resource = null;
	this.context = null;
	this.parent = null;
	this.scene = null;
	this.layer = null;
	this.childs = {};
	
	/*
	 * properties
	 */
	this.style = null;
	this.alpha = 1;
	
	this.centerX = function(){
		if(this.orgType == 1){
			return this.x;
		}else{
			return this.x + this.width / 2;
		}
	};
	
	this.centerY = function(){
		if(this.orgType == 1){
			return this.y;
		}else{
			return this.y + this.height / 2;
		}
	};
	
	this.getXByCenterX = function(cx){
		if(this.orgType == 1){
			return cx;
		}else{
			return cx - this.width / 2;
		}
	};
	
	this.getYByCenterY = function(cy){
		if(this.orgType == 1){
			return cy;
		}else{
			return cy - this.height / 2;
		}
	};
	
	this.addchild = function(spirit){
		if(spirit != null){
			spirit.context = this.context;
			spirit.parent = this;
			spirit.scene = this.scene;
			spirit.layer = this.layer;
			if(spirit.position == 0){
				spirit.x = this.centerX() + spirit.relativeX;
				spirit.y = this.centerY() + spirit.relativeY;
			}
			this.scene.childs.push(spirit);
			if(spirit.substanceType==1){
				this.scene.rigidBodyChilds.push(spirit);
			}
			this.childs[spirit.name] = spirit;
		}
	}
	
	this.runAction = function(action){
		action();
	}
	
	this.eventListener = function(event){
		
	}
	
	this.hitTest = function(hitSpirit,result){

	}
	
	this.run = function(){
		this.update();
		this.draw();
		/**
		 * childs run
		 */
		for(var key in this.childs){
			this.childs[key].run();
		}
	}
	
	this.fixedPosition = function(){
		if(this.isFollowParent == 0 && this.parent != null){
			this.rotate = this.parent.rotate + this.relativeRotate;
			if(this.position == 0){
				if(this.parent.orgType == 0){
					this.x = this.parent.centerX() + this.relativeX;
					this.y = this.parent.centerY() + this.relativeY;
				}else{
					this.x = this.parent.x + this.relativeX;
					this.y = this.parent.y + this.relativeY;
				}
			}
		}
	}
	
	this.draw = function(){
		
		/**
		 * fixed parent position
		 */
		this.fixedPosition();
		
		switch (this.spiritType) {
		case 0:
			
			switch (this.orgType) {
			case 0:
				Engine2D.util.drawRect2(this);
				break;
			case 1:
				Engine2D.util.drawCircles(this);
				break;
			case 2:
				Engine2D.util.drawLine(this);
				break;
			case 3:
				Engine2D.util.drawText(this);
				break;
			}
			
			break;
		case 1:
			
			break;
		case 2:
	
			break;
		}
		
	}
	
	this.update = function(){

	}
	
}

Engine2D.action = {};

Engine2D.action.moveTo = function(spirit ,x ,y ,speed){
	var disX = spirit.x - x;
	var disY = spirit.y - y;
	var moveX = 0;
	var moveY = 0;
	if(disX>0){
		if(speed>disX){
			moveX = - disX; 
		}else{
			moveX = - speed;
		}
	}else{
		if(-speed<disX){
			moveX = - disX; 
		}else{
			moveX = speed;
		}
	}
	if(disY>0){
		if(speed>disY){
			moveY = - disY; 
		}else{
			moveY = - speed;
		}
	}else{
		if(-speed<disY){
			moveY = - disY; 
		}else{
			moveY = speed;
		}
	}
	Engine2D.action.move(spirit,moveX,moveY);
}

Engine2D.action.move = function(spirit,moveX,moveY){
	spirit.x += moveX;
	spirit.y += moveY;
}

Engine2D.action.forward = function(spirit,distance,isBoundary){
	
	var moveX = Math.floor(Engine2D.util.cos(spirit.rotate) * distance);
	var moveY = Math.floor(Engine2D.util.sin(spirit.rotate) * distance);
	
	if(isBoundary){
		
		if((spirit.x + moveX + spirit.width)>=spirit.scene.instance.width || (spirit.x + moveX - spirit.width) <= 0){
			moveX = 0;
		}
		
		if((spirit.y + moveY + spirit.height)>=spirit.scene.instance.height || (spirit.y + moveY - spirit.height) <= 0){
			moveY = 0;
		}
		
	}
	
	spirit.x += moveX;
	spirit.y += moveY;

}

Engine2D.action.back = function(spirit,distance,isBoundary){
	
	var moveX = Engine2D.util.cos(spirit.rotate) * distance;
	var moveY = Engine2D.util.sin(spirit.rotate) * distance;
	
	if(isBoundary){
		
		if((spirit.x + moveX + spirit.width)>=spirit.scene.instance.width || (spirit.x + moveX - spirit.width) <= 0){
			moveX = 0;
		}
		
		if((spirit.y + moveY + spirit.height)>=spirit.scene.instance.height || (spirit.y + moveY - spirit.height) <= 0){
			moveY = 0;
		}
		
	}
	
	spirit.x -= moveX;
	spirit.y -= moveY;
	
}

Engine2D.action.stop = function(spirit){
	
	spirit.x = spirit.x;
	spirit.y = spirit.y;
	
}

Engine2D.action.rotate = function(spirit,rotate){
	spirit.rotate += rotate;
}

Engine2D.action.rotateTo = function(spirit,rotate,speed){
	if(spirit.rotate!=rotate){
		if(spirit.rotate>rotate){
			spirit.rotate -= speed;
		}else{
			spirit.rotate += speed;
		}
	}
}

Engine2D.util = {};

Engine2D.util.drawText = function(spirit){
	spirit.context.fillStyle = spirit.style;//填充颜色,默认是黑色
	spirit.context.globalAlpha = spirit.alpha;//透明度
	if(spirit.font == null){
		spirit.font = "10px Georgia"
	}
	spirit.context.font = spirit.font;
	spirit.context.fillText(spirit.name,spirit.x,spirit.y);
}

Engine2D.util.drawPoint = function(spirit,x,y,r,color,alpha){
	spirit.context.beginPath();
	spirit.context.fillStyle = color;//填充颜色,默认是黑色
	spirit.context.globalAlpha = alpha;//透明度
	spirit.context.arc(x,y,r,0,360,false);
	spirit.context.fill();//画实心圆
	spirit.context.closePath();
}

Engine2D.util.drawCircles = function(spirit){

	spirit.context.beginPath();
	spirit.context.fillStyle = spirit.style;//填充颜色,默认是黑色
	spirit.context.globalAlpha = spirit.alpha;//透明度
	spirit.context.arc(spirit.x,spirit.y,spirit.width,0,360,false);
	spirit.context.fill();//画实心圆
	spirit.context.closePath();
	if(spirit.showName){
		Engine2D.util.showName(spirit);
	}
}

Engine2D.util.drawCirclesArray = function(spirits){
	spirit.context.beginPath();
	for(var i = 0;i<spirits.length;i++){
		let spirit = spirits[i];
		spirit.context.fillStyle = spirit.style;//填充颜色,默认是黑色
		spirit.context.globalAlpha = spirit.alpha;//透明度
		spirit.context.arc(spirit.x,spirit.y,spirit.width,0,360,false);
		spirit.context.fill();//画实心圆
	}
	spirit.context.closePath();
}

Engine2D.util.drawRect = function(spirit){
	spirit.context.save();
	spirit.context.translate(spirit.x/2,spirit.y/2);
	spirit.context.rotate(spirit);
	spirit.context.scale(0.5,0.5);
	spirit.context.fillStyle = spirit.style;//填充颜色,默认是黑色
	spirit.context.globalAlpha = spirit.alpha;//透明度
	spirit.context.fillRect(spirit.x,spirit.y,spirit.width,spirit.height);
	spirit.context.restore();
}

Engine2D.util.drawRect2 = function(spirit){
	var leftTop = [spirit.x, spirit.y];
	var rightTop = [spirit.x + spirit.width, spirit.y];
	var rightBottom = [spirit.x + spirit.width, spirit.y + spirit.height];
	var leftBottom = [spirit.x, spirit.y + spirit.height];
	var angleOfRad = Engine2D.engine.Vec2.degToRad(spirit.rotate);

	var rotateLeftTop = Engine2D.engine.Vec2.rotatePoint([spirit.centerX(), spirit.centerY()], leftTop, angleOfRad);
	var rotateRightTop = Engine2D.engine.Vec2.rotatePoint([spirit.centerX(), spirit.centerY()], rightTop, angleOfRad);
	var rotateRightBottom = Engine2D.engine.Vec2.rotatePoint([spirit.centerX(), spirit.centerY()], rightBottom, angleOfRad);
	var rotateLeftBottom = Engine2D.engine.Vec2.rotatePoint([spirit.centerX(), spirit.centerY()], leftBottom, angleOfRad);
	
	spirit.context.beginPath();
	spirit.context.globalAlpha = spirit.alpha;//透明度
	spirit.context.moveTo(rotateLeftTop.x, rotateLeftTop.y);
	spirit.context.lineTo(rotateRightTop.x, rotateRightTop.y);
	spirit.context.lineTo(rotateRightBottom.x, rotateRightBottom.y);
	spirit.context.lineTo(rotateLeftBottom.x, rotateLeftBottom.y);
	spirit.context.closePath();
	
	spirit.context.save();
	spirit.context.fillStyle = spirit.style;
	spirit.context.fill();
	spirit.context.restore();
}

/**
 * x1=x+s·cosθ
 * y1=y+s·sinθ
 */
Engine2D.util.drawLine = function(spirit){
	spirit.context.beginPath();
	spirit.context.moveTo(spirit.x,spirit.y);
	spirit.context.strokeStyle = spirit.style;//填充颜色,默认是黑色
	spirit.context.globalAlpha = spirit.alpha;//透明度
	spirit.context.lineTo(spirit.x + Engine2D.util.cos(spirit.rotate) * spirit.width,spirit.y + Engine2D.util.sin(spirit.rotate) * spirit.width);
	spirit.context.stroke();
}

/**
 * x1=x+s·cosθ
 * y1=y+s·sinθ
 */
Engine2D.util.drawLine2 = function(spirit,point,rotate,dis,color,alpha){
	spirit.context.beginPath();
	spirit.context.moveTo(point.x,point.y);
	spirit.context.strokeStyle = color;//填充颜色,默认是黑色
	spirit.context.globalAlpha = alpha;//透明度
	spirit.context.lineTo(point.x + Engine2D.util.cos(rotate) * dis,point.y + Engine2D.util.sin(rotate) * dis);
	spirit.context.stroke();
}

Engine2D.util.sin = function(angle){
	return Math.sin(angle * Math.PI / 180);
}

Engine2D.util.cos = function(angle){
	return Math.cos(angle * Math.PI / 180);
}

Engine2D.util.showName = function(spirit){
	if(spirit.font == null){
		spirit.font = "10px Georgia"
	}
	spirit.context.font = spirit.font;
	spirit.context.fillText(spirit.name,spirit.x,spirit.y);
}

Engine2D.util.randomColor = function (){
    var colorValue = "0,1,2,3,4,5,6,7,8,9,a,b,c,d,e,f";
    var colorArray = colorValue.split(",");
    var color = "#";
    for( var i = 0; i < 6; i++ ){
        color += colorArray[ Math.floor( Math.random() * 16 ) ];
    }
    return color;
}

Engine2D.engine = {};

Engine2D.engine.afterHitTest = function(scene){
	if(scene!=null && scene.afterHitTest!=null){
		scene.afterHitTest();
	}
}

Engine2D.engine.hitTest = function(scene){
	
	if(scene!=null){
		
		for(var i = 0;i<scene.rigidBodyChilds.length;i++){
			
			for(var j = 0;j<scene.rigidBodyChilds.length;j++){
				if(i != j){
					
					if(scene.rigidBodyChilds[i].parent!=undefined && scene.rigidBodyChilds[i].parent!=null
							&& scene.rigidBodyChilds[j].parent!=undefined && scene.rigidBodyChilds[j].parent!=null
							&& scene.rigidBodyChilds[i].parent.name == scene.rigidBodyChilds[i].parent.name){
						break;
					}
					
					if(scene.rigidBodyChilds[i].parent!=undefined && scene.rigidBodyChilds[i].parent!=null){

						if(scene.rigidBodyChilds[i].parent.name != scene.rigidBodyChilds[j].name){
							
							let result = Engine2D.engine.typeHitTest(scene.rigidBodyChilds[i],scene.rigidBodyChilds[j]);
							
							if(result){
								Engine2D.engine.objHitTest(scene.rigidBodyChilds[i],scene.rigidBodyChilds[j],result);
							}
							
						}
						
					}else{
						
						if(scene.rigidBodyChilds[j].parent!=undefined && scene.rigidBodyChilds[j].parent!=null){
							
							if(scene.rigidBodyChilds[j].parent.name != scene.rigidBodyChilds[i].name){
								
								let result = Engine2D.engine.typeHitTest(scene.rigidBodyChilds[i],scene.rigidBodyChilds[j]);

								if(result){
									Engine2D.engine.objHitTest(scene.rigidBodyChilds[i],scene.rigidBodyChilds[j],result);
								}
								
							}
							
						}else{
							
							let result = Engine2D.engine.typeHitTest(scene.rigidBodyChilds[i],scene.rigidBodyChilds[j]);
							
							if(result){
								Engine2D.engine.objHitTest(scene.rigidBodyChilds[i],scene.rigidBodyChilds[j],result);
//								scene.rigidBodyChilds[i].hitTest(scene.rigidBodyChilds[j],result);
							}
							
						}
						
					}
					
				}
			}
		}
		
	}
	
}

Engine2D.engine.objHitTest = function(rigidBody1,rigidBody2,result){
	rigidBody1.hitTest(rigidBody2,result);
	rigidBody2.hitTest(rigidBody1,result);
}

Engine2D.engine.mouseListen = function(instance){
	
	instance.canvas.addEventListener('click',function(event){
		instance.mouseX = event.clientX - canvas.getBoundingClientRect().left;
		instance.mouseY = event.clientY - canvas.getBoundingClientRect().top;
		//console.log(instance.mouseX + ":" + instance.mouseY);
		instance.click();
	});
	
	instance.canvas.addEventListener('mousemove',function(event){
		instance.mouseX = event.clientX - canvas.getBoundingClientRect().left;
		instance.mouseY = event.clientY - canvas.getBoundingClientRect().top;
		//console.log(instance.mouseX + ":" + instance.mouseY);
		instance.mousemove();
	});
	
	instance.canvas.addEventListener('mousedown',function(event){
		instance.mouseX = event.clientX - canvas.getBoundingClientRect().left;
		instance.mouseY = event.clientY - canvas.getBoundingClientRect().top;
		//console.log(instance.mouseX + ":" + instance.mouseY);
		instance.mousedown();
	});
	
	instance.canvas.addEventListener('mouseup',function(event){
		instance.mouseX = event.clientX - canvas.getBoundingClientRect().left;
		instance.mouseY = event.clientY - canvas.getBoundingClientRect().top;
		//console.log(instance.mouseX + ":" + instance.mouseY);
		instance.mouseup();
	});
	
}

Engine2D.engine.typeHitTest = function(a, b) {
	
	/**
	 * orgType 0:fillRect,1:circles,2:line
	 */
	if(a.orgType == 0 && b.orgType == 0){  //fillRect to fillRect
		
		return Engine2D.engine.RectToRectTest(a,b);
		
	}else if(a.orgType == 1 && b.orgType == 1){  //circles to circles
		
		return Engine2D.engine.circlesCollisionTest(a,b);
		
	}else if(a.orgType == 2 && b.orgType == 2){  //line to line
		
	}else if((a.orgType == 1 && b.orgType == 0) || (a.orgType == 0 && b.orgType == 1)){  //fillRect to circles

		return Engine2D.engine.YDCollision(a,b);
		
	}else if((a.orgType == 1 && b.orgType == 2) || (a.orgType == 2 && b.orgType == 1)){  //circles to line
		//x,y,r,x1,y1,x2,y2
		var x = 0;
		var y = 0;
		var r = 0;
		var x1 = 0;
		var y1 = 0;
		var x2 = 0;
		var y2 = 0;
		if(a.orgType == 1){
			x = a.x;
			y = a.y;
			r = a.width;
			x1 = b.x;
			y1 = b.y;
			x2 = b.x + Engine2D.util.cos(b.rotate) * b.width;
			y2 = b.y + Engine2D.util.sin(b.rotate) * b.width;
		}else{
			x = b.x;
			y = b.y;
			r = b.width;
			x1 = a.x;
			y1 = a.y;
			x2 = a.x + Engine2D.util.cos(a.rotate) * a.width;
			y2 = a.y + Engine2D.util.sin(a.rotate) * a.width;
		}
		
		return Engine2D.engine.circlesToLineCollisionTest(x,y,r,x1,y1,x2,y2);

	}else if((a.orgType == 0 && b.orgType == 2) || (a.orgType == 2 && b.orgType == 0)){  //fillRect to line
		
		if(a.orgType == 0){
			return Engine2D.engine.RectToLineTestV2(a,b);
		}else{
			return Engine2D.engine.RectToLineTestV2(b,a);
		}
		
	}
	
	return false;
}

Engine2D.engine.Vec2 = function(x, y) {
	this.x = x;
	this.y = y;
}

Engine2D.engine.Vec2.distance = function (v1, v2) {
	var dx = v1.x - v2.x;
	var	dy = v1.y - v2.y;
	return Math.sqrt(dx * dx + dy * dy);
}

Engine2D.engine.Vec2.distance2 = function(x1, y1, x2, y2) {
	return Math.sqrt(Math.pow(x2 - x1, 2) + Math.pow(y2 - y1, 2))
}

Engine2D.engine.Vec2.degToRad = function (deg) {
	return deg * Math.PI / 180;
}

Engine2D.engine.Vec2.add = function (v1, v2) {
	return new Engine2D.engine.Vec2(v1.x + v2.x, v1.y + v2.y);
}

Engine2D.engine.Vec2.substract = function (v1, v2) {
	return new Engine2D.engine.Vec2(v1.x - v2.x, v1.y - v2.y);
}

Engine2D.engine.Vec2.dot = function (v1, v2) {
	return v1.x * v2.x + v1.y * v2.y;
}


Engine2D.engine.Vec2.rotatePoint = function (pivot, point, angle) {
	// Rotate clockwise, angle in radians
	var x = Math.round((Math.cos(angle) * (point[0] - pivot[0])) -(Math.sin(angle) * (point[1] - pivot[1])) +pivot[0]);
	var y = Math.round((Math.sin(angle) * (point[0] - pivot[0])) + (Math.cos(angle) * (point[1] - pivot[1])) + pivot[1]);
	return { x: x, y: y};
}

Engine2D.engine.Vec2.prototype = {
	length : function () {
		return Math.sqrt(this.x * this.x + this.y * this.y);
	},

	normalize : function () {
		var l = this.length();
		return new Engine2D.engine.Vec2(this.x / l, this.y / l);
	},

	normL : function () {
		return new Engine2D.engine.Vec2(this.y, -this.x);
	}
};

Engine2D.engine.circlesCollisionRectTest = function(a,b){
	var jw = 0;
	var jh = 0;
	var rr = 0;
	var jx = 0;
	var jy = 0;
	var rx = 0;
	var ry = 0;
	if(a.orgType == 1){
		rx = a.x;
		ry = a.y;
		rr = a.width;
		jx = b.x/2;
		jy = b.y/2;
		jw = b.width;
		jh = b.height;
	}else{
		rx = b.x;
		ry = b.y;
		rr = b.width;
		jx = a.x/2;
		jy = a.y/2;
		jw = a.width;
		jh = a.height;
	}
	var _rx = rx - jx;
	var _ry = ry - jy;
	var dx = Math.min(_rx, jw * 0.5);
	var dx1 = Math.max(dx, -jw * 0.5);
	var dy = Math.min(_ry, jh * 0.5);
	var dy1 = Math.max(dy, -jh * 0.5);
	return (dx1 - _rx) * (dx1 - _rx) + (dy1 - _ry) *  (dy1 - _ry) <= rr * rr;
}

Engine2D.engine.IsCirlceCollisionRect = function(a, b){
	var jw = 0;
	var jh = 0;
	var rr = 0;
	var jx = 0;
	var jy = 0;
	var rx = 0;
	var ry = 0;
	if(a.orgType == 1){
		rx = a.x;
		ry = a.y;
		rr = a.width;
		jx = b.x;
		jy = b.y;
		jw = b.width;
		jh = b.height;
	}else{
		rx = b.x;
		ry = b.y;
		rr = b.width;
		jx = a.x;
		jy = a.y;
		jw = a.width;
		jh = a.height;
	}
	
    var arcR  = rr;
    var arcOx = rx;
    var arcOy = ry;

    //分别判断矩形4个顶点与圆心的距离是否<=圆半径；如果<=，说明碰撞成功   
    if(((jx-arcOx) * (jx-arcOx) + (jy-arcOy) * (jy-arcOy)) <= arcR * arcR)   
        return true;   
    if(((jx+jw-arcOx) * (jx+jw-arcOx) + (jy-arcOy) * (jy-arcOy)) <= arcR * arcR)   
        return true;   
    if(((jx-arcOx) * (jx-arcOx) + (jy+jh-arcOy) * (jy+jh-arcOy)) <= arcR * arcR)   
        return true;   
    if(((jx+jw-arcOx) * (jx+jw-arcOx) + (jy+jh-arcOy) * (jy+jh-arcOy)) <= arcR * arcR)   
        return true;

    //判断当圆心的Y坐标进入矩形内时X的位置，如果X在(rectX-arcR)到(rectX+rectW+arcR)这个范围内，则碰撞成功   
    var minDisX = 0;   
    if(arcOy >= jy && arcOy <= jy + jh)
    {   
        if(arcOx < jx)   
            minDisX = jx - arcOx;   
        else if(arcOx > jx + jw)   
            minDisX = arcOx - jx - jw;   
        else    
            return true;   
        if(minDisX <= arcR)   
            return true;   
    }

    //判断当圆心的X坐标进入矩形内时Y的位置，如果X在(rectY-arcR)到(rectY+rectH+arcR)这个范围内，则碰撞成功
    var minDisY = 0;   
    if(arcOx >= jx && arcOx <= jx + jw)
    {   
        if(arcOy < jy)   
            minDisY = jy - arcOy;   
        else if(arcOy > jy + jh)   
            minDisY = arcOy - jy - jh;   
        else  
            return true;   
        if(minDisY <= arcR)   
            return true;   
    }

    return false; 
}

Engine2D.engine.YDCollision = function(a,b){
	
	if(a.orgType == 1){
		return Engine2D.engine.YdetectCollision(b,a);
	}else{
		return Engine2D.engine.YdetectCollision(a,b);
	}
	
}

Engine2D.engine.YdetectCollision = function(rect, circle) {
	var cx, cy;
	var angleOfRad = Engine2D.engine.Vec2.degToRad(-rect.rotate);
	var rectCenterX = rect.x + (rect.width / 2.0);
	var rectCenterY = rect.y + (rect.height / 2.0);

	var rotateCircleX = (Math.cos(angleOfRad) * (circle.x - rectCenterX)) - (Math.sin(angleOfRad) * (circle.y - rectCenterY)) + rectCenterX;
	var rotateCircleY = Math.sin(angleOfRad) * (circle.x - rectCenterX) + Math.cos(angleOfRad) * (circle.y - rectCenterY) + rectCenterY;
	
	if (rotateCircleX < rect.x) {
		cx = rect.x;
	} else if (rotateCircleX > rect.x + rect.width) {
		cx = rect.x + rect.width;
	} else {
		cx = rotateCircleX;
	}

	if (rotateCircleY < rect.y) {
		cy = rect.y;
	} else if (rotateCircleY > rect.y + rect.height) {
		cy = rect.y + rect.height;
	} else {
		cy = rotateCircleY;
	}

	if (Engine2D.engine.Vec2.distance2(rotateCircleX, rotateCircleY, cx, cy) < circle.width) {
		return true;
	}

	return false;
}

Engine2D.engine.circlesCollisionTest = function(a,b){
	var axis = new Engine2D.engine.Vec2(a.x - b.x, a.y - b.y);
	var proA = Engine2D.engine.Vec2.dot(new Engine2D.engine.Vec2(a.x, a.y), axis) / axis.length();
	var projectionA = {min : proA - a.width, max : proA + a.width};
	var proB = Engine2D.engine.Vec2.dot(new Engine2D.engine.Vec2(b.x, b.y), axis) / axis.length();
	var projectionB = {min : proB - b.width, max : proB + b.width};
	if (Engine2D.engine.isOverlay(projectionA, projectionB)) {
		return false;
	}
	return true;
}

//圆与线段碰撞检测
//圆心p(x, y), 半径r, 线段两端点p1(x1, y1)和p2(x2, y2)
Engine2D.engine.circlesToLineCollisionNewTest = function(x,y,r,x1,y1,x2,y2){
	
	var disP1 = Math.sqrt((x - x1) * (x - x1) + (y - y1) * (y - y1));
	var disP2 = Math.sqrt((x - x2) * (x - x2) + (y - y2) * (y - y2));
	
	var axisX1 = x1;
	var axisY1 = y1;
	
	if(disP1>disP2){
		axisX1 = x2;
		axisY1 = y2;
	}
	
	var axisX = x - axisX1;
	var axisY = y - axisY1;
	
	var axis = new Engine2D.engine.Vec2(axisX, axisY);
	
	var proA = Engine2D.engine.Vec2.dot(new Engine2D.engine.Vec2(axisX1, axisY1), axis) / axis.length();
	var projectionA = {min : proA, max : proA};
	var proB = Engine2D.engine.Vec2.dot(new Engine2D.engine.Vec2(x, y), axis) / axis.length();
	var projectionB = {min : proB - r, max : proB + r};
	
	if (Engine2D.engine.isOverlay(projectionA, projectionB)) {
		return false;
	}
	
	return true;
}

Engine2D.engine.RectToLineTestV2 = function(a,b){
	var leftTop = [a.x, a.y];
	var rightTop = [a.x + a.width, a.y];
	var rightBottom = [a.x + a.width, a.y + a.height];
	var leftBottom = [a.x, a.y + a.height];
	var angleOfRad = Engine2D.engine.Vec2.degToRad(a.rotate);

	var rotateLeftTop = Engine2D.engine.Vec2.rotatePoint([a.centerX(), a.centerY()], leftTop, angleOfRad);
	var rotateRightTop = Engine2D.engine.Vec2.rotatePoint([a.centerX(), a.centerY()], rightTop, angleOfRad);
	var rotateRightBottom = Engine2D.engine.Vec2.rotatePoint([a.centerX(), a.centerY()], rightBottom, angleOfRad);
	var rotateLeftBottom = Engine2D.engine.Vec2.rotatePoint([a.centerX(), a.centerY()], leftBottom, angleOfRad);
	
	var lineA = {x:b.x,y:b.y};
	var lineB = {x:b.x + Engine2D.util.cos(b.rotate) * b.width,y:b.y + Engine2D.util.sin(b.rotate) * b.width};

	var hitResult = false;
	
	var minDis = null;
	
	var result1 = Engine2D.engine.LineToLineTestV3(lineA,lineB,rotateLeftTop,rotateLeftBottom);

	var result2 = Engine2D.engine.LineToLineTestV3(lineA,lineB,rotateLeftTop,rotateRightTop);

	var result3 = Engine2D.engine.LineToLineTestV3(lineA,lineB,rotateRightTop,rotateRightBottom);

	var result4 = Engine2D.engine.LineToLineTestV3(lineA,lineB,rotateLeftBottom,rotateRightBottom);
	
	if(result1){
		var onceDis = Engine2D.engine.Vec2.distance(lineA,result1);
		if(minDis == null){
			minDis = onceDis;
			hitResult = result1;
		}else if(minDis > onceDis){
			minDis = onceDis;
			hitResult = result1;
		}
	}
	
	if(result2){
		var onceDis = Engine2D.engine.Vec2.distance(lineA,result2);
		if(minDis == null){
			minDis = onceDis;
			hitResult = result2;
		}else if(minDis > onceDis){
			minDis = onceDis;
			hitResult = result2;
		}
	}
	
	if(result3){
		var onceDis = Engine2D.engine.Vec2.distance(lineA,result3);
		if(minDis == null){
			minDis = onceDis;
			hitResult = result3;
		}else if(minDis > onceDis){
			minDis = onceDis;
			hitResult = result3;
		}
	}
	
	if(result4){
		var onceDis = Engine2D.engine.Vec2.distance(lineA,result4);
		if(minDis == null){
			minDis = onceDis;
			hitResult = result4;
		}else if(minDis > onceDis){
			minDis = onceDis;
			hitResult = result4;
		}
	}
	
	return hitResult;
}

Engine2D.engine.RectToLineTest = function(a,b){
	var leftTop = [a.x, a.y];
	var rightTop = [a.x + a.width, a.y];
	var rightBottom = [a.x + a.width, a.y + a.height];
	var leftBottom = [a.x, a.y + a.height];
	var angleOfRad = Engine2D.engine.Vec2.degToRad(a.rotate);

	var rotateLeftTop = Engine2D.engine.Vec2.rotatePoint([a.centerX(), a.centerY()], leftTop, angleOfRad);
	var rotateRightTop = Engine2D.engine.Vec2.rotatePoint([a.centerX(), a.centerY()], rightTop, angleOfRad);
	var rotateRightBottom = Engine2D.engine.Vec2.rotatePoint([a.centerX(), a.centerY()], rightBottom, angleOfRad);
	var rotateLeftBottom = Engine2D.engine.Vec2.rotatePoint([a.centerX(), a.centerY()], leftBottom, angleOfRad);
	
	var x1 = b.x;
	var y1 = b.y;
	var x2 = b.x + Engine2D.util.cos(b.rotate) * b.width;
	var y2 = b.y + Engine2D.util.sin(b.rotate) * b.width;
	
	var result = false;
	
	result = Engine2D.engine.LineToLineTest(x1,y1,x2,y2,rotateLeftTop.x,rotateLeftTop.y,rotateLeftBottom.x,rotateLeftBottom.y);
	
	if(!result){
		result = Engine2D.engine.LineToLineTest(x1,y1,x2,y2,rotateLeftTop.x,rotateLeftTop.y,rotateRightTop.x,rotateRightTop.y);
	}
	
	if(!result){
		result = Engine2D.engine.LineToLineTest(x1,y1,x2,y2,rotateRightTop.x,rotateRightTop.y,rotateRightBottom.x,rotateRightBottom.y);
	}
	
	if(!result){
		result = Engine2D.engine.LineToLineTest(x1,y1,x2,y2,rotateLeftBottom.x,rotateLeftBottom.y,rotateRightBottom.x,rotateRightBottom.y);
	}
		
	return result;
}

Engine2D.engine.RectToRectTest = function(a,b){

	var sidesA = Engine2D.engine.getRectSides(a);
	var sidesB = Engine2D.engine.getRectSides(b);

	var result = false;
	
	for(var i=0;i<sidesA.length;i++){
		
		var sideA = sidesA[i];
		
		for(var j=0;j<sidesB.length;j++){
			
			var sideB = sidesB[j];
			
			result = Engine2D.engine.LineToLineTestV4(sideA,sideB);

			if(result){
				return result;
			}

		}

	}
	
	return result;
}

Engine2D.engine.getRectSides = function(r){
	var leftTop = [r.x, r.y];
	var rightTop = [r.x + r.width, r.y];
	var rightBottom = [r.x + r.width, r.y + r.height];
	var leftBottom = [r.x, r.y + r.height];
	var angleOfRad = Engine2D.engine.Vec2.degToRad(r.rotate);

	var rotateLeftTop = Engine2D.engine.Vec2.rotatePoint([r.centerX(), r.centerY()], leftTop, angleOfRad);
	var rotateRightTop = Engine2D.engine.Vec2.rotatePoint([r.centerX(), r.centerY()], rightTop, angleOfRad);
	var rotateRightBottom = Engine2D.engine.Vec2.rotatePoint([r.centerX(), r.centerY()], rightBottom, angleOfRad);
	var rotateLeftBottom = Engine2D.engine.Vec2.rotatePoint([r.centerX(), r.centerY()], leftBottom, angleOfRad);
	
	var sides = new Array();
	sides.push([rotateLeftTop.x,rotateLeftTop.y,rotateLeftBottom.x,rotateLeftBottom.y]);
	sides.push([rotateLeftTop.x,rotateLeftTop.y,rotateRightTop.x,rotateRightTop.y]);
	sides.push([rotateRightTop.x,rotateRightTop.y,rotateRightBottom.x,rotateRightBottom.y]);
	sides.push([rotateLeftBottom.x,rotateLeftBottom.y,rotateRightBottom.x,rotateRightBottom.y]);
	return sides;
}

Engine2D.engine.LineToLineTestV2 = function(lineA,lineB){
	var x_start_1 = lineA[0];
	var	y_start_1 = lineA[1]; 
	var	x_end_1 = lineA[2];
	var y_end_1 = lineA[3];
	var x_start_2 = lineB[0];
	var y_start_2 = lineB[1]; 
	var x_end_2 = lineB[2];
	var y_end_2 = lineB[3];
	var temp = ((y_end_2-y_start_2)*(x_end_1-x_start_1) - (x_end_2-x_start_2)*(y_end_1-y_start_1));
	var t1 = ((x_end_2-x_start_2)*(y_start_1-y_start_2) - (y_end_2-y_start_2)*(x_start_1-x_start_2)) / temp;
	var t2 = ((x_end_1-x_start_1)*(y_start_1-y_start_2) - (y_end_1-y_start_1)*(x_start_1-x_start_2)) / temp;

    if (t1 >= 0 && t1 <= 1 && t2 >= 0 && t2 <= 1) {
       return {point:{x:t1*x_end_1,y:t1*y_end_1}};
    }
    
    return false;
}

Engine2D.engine.LineToLineTest = function(x_start_1,y_start_1,x_end_1,y_end_1,x_start_2,y_start_2,x_end_2,y_end_2){
	var temp = ((y_end_2-y_start_2)*(x_end_1-x_start_1) - (x_end_2-x_start_2)*(y_end_1-y_start_1));
	var t1 = ((x_end_2-x_start_2)*(y_start_1-y_start_2) - (y_end_2-y_start_2)*(x_start_1-x_start_2)) / temp;
	var t2 = ((x_end_1-x_start_1)*(y_start_1-y_start_2) - (y_end_1-y_start_1)*(x_start_1-x_start_2)) / temp;
	
	var denominator = (b.y - a.y)*(d.x - c.x) - (a.x - b.x)*(c.y - d.y);  
	
    if (t1 >= 0 && t1 <= 1 && t2 >= 0 && t2 <= 1) {

      return {point:{x:t1*x_end_1,y:t1*y_end_1}};
    }
    
    return false;
}

Engine2D.engine.LineToLineTestV3 = function segmentsIntr(a, b, c, d){  
	  
    // 三角形abc 面积的2倍  
    var area_abc = (a.x - c.x) * (b.y - c.y) - (a.y - c.y) * (b.x - c.x);  
  
    // 三角形abd 面积的2倍  
    var area_abd = (a.x - d.x) * (b.y - d.y) - (a.y - d.y) * (b.x - d.x);   
  
    // 面积符号相同则两点在线段同侧,不相交 (对点在线段上的情况,本例当作不相交处理);  
    if ( area_abc*area_abd>=0 ) {  
        return false;  
    }  
  
    // 三角形cda 面积的2倍  
    var area_cda = (c.x - a.x) * (d.y - a.y) - (c.y - a.y) * (d.x - a.x);  
    // 三角形cdb 面积的2倍  
    // 注意: 这里有一个小优化.不需要再用公式计算面积,而是通过已知的三个面积加减得出.  
    var area_cdb = area_cda + area_abc - area_abd ;  
    if (  area_cda * area_cdb >= 0 ) {  
        return false;  
    }  
  
    //计算交点坐标  
    var t = area_cda / ( area_abd- area_abc );  
    var dx= t*(b.x - a.x),  
        dy= t*(b.y - a.y);  
    return { x: a.x + dx , y: a.y + dy };  
}

Engine2D.engine.LineToLineTestV4 = function(lineA,lineB){
	var a = {x:lineA[0],y:lineA[1]};
	var b = {x:lineA[2],y:lineA[3]};
	var c = {x:lineB[0],y:lineB[1]};
	var d = {x:lineB[2],y:lineB[3]};
	return Engine2D.engine.LineToLineTestV3(a,b,c,d);
}

//圆与线段碰撞检测
//圆心p(x, y), 半径r, 线段两端点p1(x1, y1)和p2(x2, y2)
Engine2D.engine.circlesToLineCollisionTest = function(x,y,r,x1,y1,x2,y2){
	var vx1 = x - x1;
	var vy1 = y - y1;
	var vx2 = x2 - x1;
	var vy2 = y2 - y1;
	
	// len = v2.length()
	var len = Math.sqrt(vx2 * vx2 + vy2 * vy2);

	// v2.normalize()
	vx2 /= len;
	vy2 /= len;

	// u = v1.dot(v2)
	// u is the vector projection length of vector v1 onto vector v2.
	var u = vx1 * vx2 + vy1 * vy2;

	// determine the nearest point on the lineseg
	var x0 = 0;
	var y0 = 0;
	if (u <= 0){
		// p is on the left of p1, so p1 is the nearest point on lineseg
		x0 = x1;
		y0 = y1;
	}else if (u >= len){
		// p is on the right of p2, so p2 is the nearest point on lineseg
		x0 = x2;
		y0 = y2;
	}else{
		// p0 = p1 + v2 * u
		// note that v2 is already normalized.
		x0 = x1 + vx2 * u;
		y0 = y1 + vy2 * u;
	}

	return (x - x0) * (x - x0) + (y - y0) * (y - y0) <= r * r;
}

Engine2D.engine.isOverlay = function(proA, proB){
	if (proA.min < proB.min) {
		min = proA.min;
	} else {
		min = proB.min;
	}

	if (proA.max > proB.max) {
		max = proA.max;
	} else {
		max = proB.max;
	}
	
	return (proA.max - proA.min) + (proB.max - proB.min) < max - min;
}

Array.prototype.remove=function(dx) {
	
	if(isNaN(dx)||dx>this.length){
		return false;
	}
	
	for(var i=0,n=0;i<this.length;i++) { 
	    if(this[i]!=this[dx]) { 
	      this[n++]=this[i] 
	    } 
	}
	
	this.length -= 1 
}
