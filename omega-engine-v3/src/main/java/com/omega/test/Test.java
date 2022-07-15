package com.omega.test;

import com.omega.engine.gpu.JNITest;
import com.omega.engine.service.impl.BusinessServiceImpl;

public class Test {
	
	static {
//		System.load("H:\\omega\\omega-ai\\omega-engine-v3\\jni\\test_cuda.dll");
	}
	
	public static void main(String[] args) {
		
//		new JNITest().max(1, 2);
		
//		BusinessServiceImpl bs = new BusinessServiceImpl();
//		bs.showImage();
//		bs.bpNetwork_iris();
//		bs.bpNetwork_mnist();
//		bs.cnnNetwork_mnist_demo();
//		bs.cnnNetwork_mnist();
//		bs.cnnNetwork_cifar10();
//		bs.cnnNetwork_vgg16_cifar10();
//		bs.vgg16_cifar10();
//		bs.alexNet_mnist();
//		bs.alexNet_cifar10();
		
		int n = 3;
		int x = 5;
		
		System.out.println((n + x) * 2);
		
		System.out.println(n * 2 + x * 2);
		
		
	}
	
}
