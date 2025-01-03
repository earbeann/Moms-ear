package com.example.baby_one_more_time

import android.content.Intent
import android.os.Bundle
import android.widget.Button
import androidx.appcompat.app.AppCompatActivity

class MyPage_Main : AppCompatActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.mypage_main)

        // 설정 버튼 찾기
        val settingsButton: Button = findViewById(R.id.settingsButton)

        // 설정 버튼 클릭 이벤트 설정
        settingsButton.setOnClickListener {
            // SettingsActivity로 이동
            val intent = Intent(this, MyPage2Activity::class.java)
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
