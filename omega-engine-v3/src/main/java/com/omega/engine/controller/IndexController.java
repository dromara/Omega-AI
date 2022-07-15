package com.omega.engine.controller;

import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;

import org.springframework.stereotype.Controller;
import org.springframework.web.bind.annotation.RequestMapping;

@Controller
public class IndexController {
	
	@RequestMapping("/")
	public String index(HttpServletRequest request,HttpServletResponse response) {
		
		return "index";
	}
	
	@RequestMapping("/AICar")
	public String aiCar(HttpServletRequest request,HttpServletResponse response) {
		
		return "2DCar/AICarDefautMap";
	}
	
	@RequestMapping("/role")
	public String role(HttpServletRequest request,HttpServletResponse response) {
		
		return "2DRole/role";
	}
	
	@RequestMapping("/mnist")
	public String mnist(HttpServletRequest request,HttpServletResponse response) {
		
		return "mnist/mnist";
	}
	
	@RequestMapping("/origin")
	public String origin(HttpServletRequest request,HttpServletResponse response) {
		
		return "origin/origin";
	}
	
	@RequestMapping("/origin2")
	public String origin2(HttpServletRequest request,HttpServletResponse response) {
		
		return "origin/origin2";
	}
	
	@RequestMapping("/training")
	public String training(HttpServletRequest request,HttpServletResponse response) {
		
		return "training/controller";
	}
	
}
