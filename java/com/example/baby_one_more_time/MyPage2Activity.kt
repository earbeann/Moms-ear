package com.example.baby_one_more_time

import android.os.Bundle
import androidx.activity.enableEdgeToEdge
import androidx.appcompat.app.AppCompatActivity
import androidx.core.view.ViewCompat
import androidx.core.view.WindowInsetsCompat
import android.widget.Button // Button 클래스를 import
import android.content.Intent
import android.widget.EditText
import android.text.method.ScrollingMovementMethod
import android.widget.Toast // 캘린더 추가
import android.widget.CalendarView // 캘린더 추가

class MyPage2Activity : AppCompatActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.mypage2) // 레이아웃 파일 설정

        val backButton: Button = findViewById<Button>(R.id.backButton)
        backButton.setOnClickListener {
            finish() // 현재 액티비티 종료하여 이전 화면으로 돌아감
        }

        // 마이 페이지로 돌아가는 MY 버튼 설정
        val myPageButton: Button = findViewById(R.id.myPageButton)
        myPageButton.setOnClickListener {
            // MainActivity로 돌아가는 Intent 생성
            val intent = Intent(this, MyPage_Main::class.java)
            // MainActivity가 스택에 하나만 존재하도록 설정
            intent.flags = Intent.FLAG_ACTIVITY_CLEAR_TOP or Intent.FLAG_ACTIVITY_SINGLE_TOP
            startActivity(intent)
        }


        // -------------------------------------------------------------------------------------------------------------------------
        //여기서부터는 하단바 설정 칸들
        // "동화" 버튼 클릭 시 Fairytale_Main으로 이동
        val fairyTaleButton = findViewById<Button>(R.id.fairyTaleButton)
        fairyTaleButton.setOnClickListener {
            val intent = Intent(this, Fairytale_Main::class.java)
            startActivity(intent)
        }

        // "home" 버튼 클릭 시 diary_main으로 이동
        val homeButton = findViewById<Button>(R.id.homeButton)
        homeButton.setOnClickListener {
            val intent = Intent(this, Diary_Main::class.java)
            startActivity(intent)
        }


        // "음악" 버튼 클릭 시 Music_Main으로 이동
        val musicButton = findViewById<Button>(R.id.musicButton)
        musicButton.setOnClickListener {
            val intent = Intent(this, Music_Main::class.java)
            startActivity(intent)
        }


        // "My" 버튼 클릭 시 MyPage_Main으로 이동
        val MYButton = findViewById<Button>(R.id.myPageButton)
        MYButton.setOnClickListener {
            val intent = Intent(this, MyPage_Main::class.java)
            startActivity(intent)
        }

    }
}