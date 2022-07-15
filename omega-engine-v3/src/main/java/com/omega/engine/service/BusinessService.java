package com.omega.engine.service;

public interface BusinessService {
	
	public void bpNetwork_iris();
	
	public void bpNetwork_mnist();
	
	public void cnnNetwork_mnist_demo();
	
	public void cnnNetwork_mnist();
	
	public void alexNet_mnist();
	
	public void cnnNetwork_cifar10();
	
	public void alexNet_cifar10();
	
	public void cnnNetwork_vgg16_cifar10();
	
	public void vgg16_cifar10();
	
	public void showImage();
	
	public void cnn_1x1();
	
	public void resnet18_mnist();
	
	public void resnet18_cifar10();
	
	public void bpNetwork_mnist(String sid,float lr);
	
	public void alexNet_mnist(String sid,float lr);
	
	public void alexNet_cifar10(String sid,float lr);
	
	public void cnnNetwork_vgg16_cifar10(String sid,float lr);
	
	public void cnnNetwork_mnist(String sid,float lr);
	
}
