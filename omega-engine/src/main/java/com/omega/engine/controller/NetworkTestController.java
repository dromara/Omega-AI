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
@RequestMapping(value = "/networkTest", method = { RequestMethod.POST, RequestMethod.GET })
public class NetworkTestController {
	
	@Autowired
	private BusinessService businessService;
	
	@RequestMapping(value = "/irisTest", method = { RequestMethod.POST, RequestMethod.GET })
	public Map<String, Object> irisTest(HttpServletRequest request, HttpServletResponse response) {
		Map<String, Object> result = new HashMap<String, Object>();
		
		try {
			
			businessService.bpNetwork_iris();
			
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
	
	@RequestMapping(value = "/mnistTest", method = { RequestMethod.POST, RequestMethod.GET })
	public Map<String, Object> mnistTest(HttpServletRequest request, HttpServletResponse response) {
		Map<String, Object> result = new HashMap<String, Object>();
		
		try {
			
			businessService.bpNetwork_mnist();
			
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
