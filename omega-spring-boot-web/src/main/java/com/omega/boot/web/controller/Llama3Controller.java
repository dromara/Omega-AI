package com.omega.boot.web.controller;

import com.omega.boot.starter.service.Llama3Service;
import com.omega.boot.web.entity.ChatDto;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.context.annotation.Lazy;
import org.springframework.web.bind.annotation.*;

/**
 * llama3接口controller
 *
 * @author haylee
 * @date 2025/05/11 14:33
 */
@RestController
@RequestMapping("/model")
public class Llama3Controller {

    @Lazy
    @Autowired
    private Llama3Service llama3Service;
    @PostMapping("/chat")
    public String chat(@RequestBody ChatDto chatDto) {
        System.out.println(chatDto.getMessage());
        return llama3Service.predict(chatDto.getMessage());
    }
}
