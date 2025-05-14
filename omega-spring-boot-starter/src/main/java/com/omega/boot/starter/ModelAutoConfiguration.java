package com.omega.boot.starter;

import org.springframework.boot.autoconfigure.AutoConfiguration;
import org.springframework.context.annotation.ComponentScan;


/**
 * 自动加载类
 *
 * @author haylee
 * @date 2025/05/11 14:33
 */
@ComponentScan(basePackages = "com.omega.boot.starter.service")
@AutoConfiguration
public class ModelAutoConfiguration {

}
