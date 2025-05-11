package com.omega.boot.starter;

import com.omega.boot.starter.service.Llama3Service;
import org.springframework.boot.autoconfigure.AutoConfiguration;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.ComponentScan;
import org.springframework.context.annotation.Import;

/**
 * 自动加载类
 *
 * @author haylee
 * @date 2025/05/11 14:33
 */
@ComponentScan(basePackages = "com.omega.boot.starter.service")
@AutoConfiguration
public class ModelAutoConfiguration {

//    @Bean
//    public Llama3Service llama23Service(){
//        return new Llama3Service();
//    }
}
