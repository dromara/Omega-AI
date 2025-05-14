package com.omega.boot.web.controller;

import com.omega.boot.starter.service.Llama3Service;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;

/**
 * 测试接口controller
 *
 * @author haylee
 * @date 2025/05/11 14:33
 */
@RestController
@RequestMapping("/model")
public class TestController {

    @Autowired
    private Llama3Service llama3Service;
    @GetMapping("/test")
    public String test(String input) {
        return "user:"+input+"\nchatbot:" + llama3Service.predict(input);
    }
}
