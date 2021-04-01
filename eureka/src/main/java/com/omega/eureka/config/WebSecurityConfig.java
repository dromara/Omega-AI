package com.omega.eureka.config;

import org.springframework.context.annotation.Configuration;
import org.springframework.security.config.annotation.web.builders.HttpSecurity;
import org.springframework.security.config.annotation.web.configuration.EnableWebSecurity;
import org.springframework.security.config.annotation.web.configuration.WebSecurityConfigurerAdapter;


@Configuration
@EnableWebSecurity
public class WebSecurityConfig extends WebSecurityConfigurerAdapter {
	
	@Override
    protected void configure(HttpSecurity http) throws Exception {
//        super.configure(http);//加这句是为了访问eureka控制台和/actuator时能做安全控制
        http.authorizeRequests()
        // 只开启eureka注册账密校验
        .antMatchers("/actuator/**").permitAll()
        // 其他请求全放过
        .anyRequest().authenticated()
        .and().csrf().disable()//关闭csrf
        // 开启基本账密校验
        .httpBasic();
    }
	
}
