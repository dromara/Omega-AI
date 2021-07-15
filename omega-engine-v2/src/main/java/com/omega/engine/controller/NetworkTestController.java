package com.omega.engine.controller;

import java.util.HashMap;
import java.util.Iterator;
import java.util.Map;

import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestMethod;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.multipart.MultipartFile;
import org.springframework.web.multipart.MultipartHttpServletRequest;
import org.springframework.web.multipart.commons.CommonsMultipartResolver;

import com.omega.common.utils.ImageUtils;
import com.omega.common.utils.LabelUtils;
import com.omega.common.utils.PrintUtils;
import com.omega.engine.database.NetworksDataBase;
import com.omega.engine.nn.data.Blob;
import com.omega.engine.nn.data.Blobs;
import com.omega.engine.nn.network.CNN;
import com.omega.engine.service.BusinessService;

@RestController
@RequestMapping(value = "/networkTest", method = { RequestMethod.POST, RequestMethod.GET })
public class NetworkTestController {
	
	@Autowired
	private BusinessService businessService;
	
	@Autowired
	private ImageUtils imageUtils;
	
	@Autowired
	private NetworksDataBase networks;
	
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
	
	@RequestMapping(value = "/mnistCNNTrain", method = { RequestMethod.POST, RequestMethod.GET })
	public Map<String, Object> mnistCNNTrain(HttpServletRequest request, HttpServletResponse response) {
		Map<String, Object> result = new HashMap<String, Object>();
		
		try {
			
			businessService.cnnNetwork_mnist();
			
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
	
	@RequestMapping(value = "/mnistCNNTest", method = { RequestMethod.POST, RequestMethod.GET })
	public Map<String, Object> mnistCNNTest(HttpServletRequest request, HttpServletResponse response) {
		Map<String, Object> result = new HashMap<String, Object>();
		
		try {
			
			//创建一个通用的多部分解析器
	        CommonsMultipartResolver multipartResolver = new CommonsMultipartResolver(request.getSession().getServletContext());
	        //判断 request 是否有文件上传,即多部分请求
	        if(multipartResolver.isMultipart(request)){
	        	
	            //转换成多部分request
	            MultipartHttpServletRequest multiRequest = (MultipartHttpServletRequest)request;
	            
	            //取得request中的所有文件名
	            Iterator<String> iter = multiRequest.getFileNames();
	            
	            while(iter.hasNext()){
	                //取得上传文件
	                MultipartFile file = multiRequest.getFile(iter.next());

	                double[][][][] imageData = imageUtils.getImageGrayPixel(file.getInputStream(),true);

	                Blob input = Blobs.blob(imageData);
	                Blob image = Blobs.transform(1, 1, 28, 28, input);

	                PrintUtils.printImage(image.maxtir[0][0]);
	                
	                if(networks.getNetworks().get("cnnMnist")!=null) {

		                CNN cnn = (CNN) networks.getNetworks().get("cnnMnist");
		                
		                Blob ouput = cnn.predict(input);
		                
		                String predict = LabelUtils.vectorTolabel(ouput.maxtir[0][0][0], new String[] {"0","1","2","3","4","5","6","7","8","9"});
		                
		                //System.out.println(JsonUtils.toJson(ouput.maxtir[0][0][0]));
		                
		                result.put("success", true);
		    			result.put("code", 200);
		    			result.put("msg", "预测成功:"+predict);
		                result.put("data", predict);
		                return result;
	                }else {
	                	result.put("success", false);
	        			result.put("code", 301);
	        			result.put("msg", "没有该该模型");
	        			return result;
	                }
	                
	            }
	            
	        }    
			
			result.put("success", false);
			result.put("code", 300);
			result.put("msg", "没有该文件");
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
