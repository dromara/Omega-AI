package com.omega.engine.controller;

import java.util.HashMap;
import java.util.Map;

import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestMethod;
import org.springframework.web.bind.annotation.RestController;

import com.omega.engine.service.BusinessService;

@RestController
@RequestMapping(value = "/tm", method = { RequestMethod.POST, RequestMethod.GET })
public class TrainingController {
	
	@Autowired
	private BusinessService businessService;
	
	@RequestMapping(value = "/start", method = { RequestMethod.POST, RequestMethod.GET })
	public Map<String, Object> start(HttpServletRequest request, HttpServletResponse response) {
		Map<String, Object> result = new HashMap<String, Object>();
		
		try {
			
			String sid = request.getParameter("sid");
			String lr = request.getParameter("lr");
			String model = request.getParameter("model");
			
			if(model != null && !model.equals("")) {
				
				float lrf = Float.valueOf(lr);
				
				switch (model) {
				case "0":
					businessService.bpNetwork_mnist(sid, lrf);
					break;
				case "1":
					businessService.cnnNetwork_mnist(sid, lrf);
					break;
				case "2":
					businessService.alexNet_mnist(sid, lrf);
					break;
				case "3":
					businessService.alexNet_cifar10(sid, lrf);
					break;
				case "4":
					businessService.cnnNetwork_vgg16_cifar10(sid, lrf);
					break;
				}
				
			}
			
			result.put("success", true);
			result.put("code", 200);
			result.put("msg", "执行成功");
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
			result.put("success", false);
			result.put("code", 400);
			result.put("msg", "系统繁忙");
		}
		return result;
    }

	@RequestMapping(value = "/updateLR", method = { RequestMethod.POST, RequestMethod.GET })
	public Map<String, Object> updateLR(HttpServletRequest request, HttpServletResponse response) {
		Map<String, Object> result = new HashMap<String, Object>();
		
		try {
			
			String sid = request.getParameter("sid");
			String lr = request.getParameter("lr");
			
			if(sid != null && lr != null) {
				float lrf = Float.valueOf(lr);
				TrainTask.updateLR(sid, lrf);
			}

			result.put("success", true);
			result.put("code", 200);
			result.put("msg", "执行成功");
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
			result.put("success", false);
			result.put("code", 400);
			result.put("msg", "系统繁忙");
		}
		return result;
    }
	
}
