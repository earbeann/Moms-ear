package com.example.baby_one_more_time

import android.content.Intent
import android.os.Bundle
import androidx.appcompat.app.AppCompatActivity
import android.widget.Button

import android.widget.CalendarView
import android.widget.TextView
import okhttp3.*
import org.json.JSONObject
import java.io.IOException
import java.util.*

class Diary2Activity : AppCompatActivity() {

    private lateinit var diaryContentTextView: TextView //다이어리

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.diary2)

        diaryContentTextView = findViewById(R.id.diaryEntry1)

        // CalendarView 설정 및 날짜 선택 리스너 추가
        val calendarView = findViewById<CalendarView>(R.id.calendarView)
        calendarView.setOnDateChangeListener { _, year, month, dayOfMonth ->
            // 날짜 선택 시 전단계로 돌아가지 않도록 실행 순서를 조정
            val selectedDate = String.format("%04d-%02d-%02d", year, month + 1, dayOfMonth)
            fetchDiaryEntry(selectedDate)
        }

        // 이전 버튼 클릭 시 Diary2Activity 종료
        val backButton = findViewById<Button>(R.id.backButton) // diary2.xml의 이전 버튼 ID가 backButton임
        backButton.setOnClickListener {
            finish() // 현재 액티비티 종료하고 이전 화면으로 돌아가기
        }

        // 하단바 설정
        setBottomNavigation()
    }


        // -------------------------------------------------------------------------------------------------------------------------
    private fun setBottomNavigation() {
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
    private fun fetchDiaryEntry(date: String) {
        val client = OkHttpClient()
        val url = "https://raw.githubusercontent.com/earbeann/mom-s_ear_repo/main/diary/$date.txt"  // GitHub의 Raw 파일 URL
        val request = Request.Builder()
            .url(url)
            .header("Cache-Control", "no-cache")  // 캐시 방지 헤더 추가
            .build()
//        val formBody = FormBody.Builder()
//            .add("date", date)  // 선택한 날짜를 POST 데이터로 추가
//            .build()

//        val request = Request.Builder()
//            .url(url)
//            .post(formBody)
//            .header("Cache-Control", "no-cache")  // 캐시 방지 헤더 추가
//            .build()
        // 일기 데이터 추가


        client.newCall(request).enqueue(object : Callback {
            override fun onFailure(call: Call, e: IOException) {
                runOnUiThread {
                    diaryContentTextView.text = "일기를 가져오는데 실패했습니다."
                }
                e.printStackTrace()
            }

            override fun onResponse(call: Call, response: Response) {
                if (response.isSuccessful) {
                    response.body?.let { responseBody ->
                        val content = responseBody.string()  // 단순 텍스트 내용 가져오기
                        runOnUiThread {
                            diaryContentTextView.text = content  // 일기 내용 표시
                        }
                    }
                } else {
                    runOnUiThread {
                        diaryContentTextView.text = "해당 날짜에 일기가 없습니다."
                    }
                }
            }
        })
    }
}

