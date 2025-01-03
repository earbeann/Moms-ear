package com.example.baby_one_more_time

import android.content.Intent
import android.os.Bundle
import android.widget.Button
import android.widget.ImageButton
import androidx.appcompat.app.AppCompatActivity
import androidx.appcompat.widget.Toolbar
import retrofit2.*
import retrofit2.converter.gson.GsonConverterFactory
import com.example.baby_one_more_time.network.ApiService
import com.example.baby_one_more_time.network.MusicRequest
import com.example.baby_one_more_time.network.MusicResponse
import android.widget.Toast
import okhttp3.*
import java.io.IOException
import android.os.Environment
import androidx.activity.enableEdgeToEdge
//추가 생성import (call callback response떄문에)
import okhttp3.Call
import okhttp3.Callback
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.Response
import retrofit2.Retrofit
import java.text.SimpleDateFormat
import java.util.Locale
import java.util.Date
import android.widget.TextView
//켈린더에 날짜지정 및 텍스트 가져오기 11_13
import android.widget.CalendarView
import java.util.Calendar

class Music_Main : AppCompatActivity() {
    private val BASE_URL = "http://192.168.101.93:8000/"
    private lateinit var apiService: ApiService
    private lateinit var diaryContentTextView: TextView
    private var selectedDate: String? = null //날짜를 선택하고 음악재생을 누르면 그날짜에 해당하는 음악 가져옴
    private var isRequestInProgress = false  // 중복 요청 방지 플래그


    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.music_main)
        diaryContentTextView = findViewById(R.id.diaryEntry1)
        // Retrofit 설정
        val retrofit = Retrofit.Builder()
            .baseUrl(BASE_URL)
            .addConverterFactory(GsonConverterFactory.create())
            .build()

        apiService = retrofit.create(ApiService::class.java)
        //켈린더에 날짜지정 및 텍스트 가져오기 11_13
        val calendarView = findViewById<CalendarView>(R.id.calendarView)
        calendarView.setOnDateChangeListener { _, year, month, dayOfMonth ->
            selectedDate = String.format("%04d-%02d-%02d", year, month + 1, dayOfMonth)
            fetchDiaryForDate(selectedDate!!)
            Toast.makeText(this, "선택된 날짜: $selectedDate", Toast.LENGTH_SHORT).show()
        }
        // 음악 생성하기 버튼 클릭 시 선택된 일기의 텍스트로 음악 생성
        val createMusicButton = findViewById<Button>(R.id.button_generate_music)
        createMusicButton.setOnClickListener {
            val diaryText = diaryContentTextView.text.toString()
            if (diaryText.isNotEmpty()) {
                generateMusicFromDiary(diaryText)
            } else {
                Toast.makeText(this, "일기 내용이 없습니다. 일기를 먼저 가져와 주세요.", Toast.LENGTH_SHORT).show()
            }
        }
        //음악실행버튼 //if절을 추가해서 날짜를 선택하고 그에 해당하는 .wav가 있으면 가져옴
        val playMusicButton = findViewById<Button>(R.id.button_play_music)
        playMusicButton.setOnClickListener {
            if (!selectedDate.isNullOrEmpty()) {
                val intent = Intent(this, Music2Activity::class.java)
                intent.putExtra("selected_date", selectedDate)
                startActivity(intent)
            } else {
                Toast.makeText(this, "먼저 날짜를 선택해 주세요.", Toast.LENGTH_SHORT).show()
            }
        }

        // Toolbar 설정
        val toolbar: Toolbar = findViewById(R.id.toolbar)
        setSupportActionBar(toolbar)
        supportActionBar?.setDisplayShowTitleEnabled(false)
        toolbar.title = ""

        // 이전 버튼 활성화 및 클릭 리스너 설정
        val backButton: ImageButton = findViewById(R.id.btn_back)
        backButton.setOnClickListener { finish() }
        // 하단바 설정
        setBottomNavigation()


    }

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

    //그날에 맞는 일기 데이터 가져오기
    private fun fetchDiaryForDate(date: String) {
        val diaryUrl =
            "https://raw.githubusercontent.com/earbeann/mom-s_ear_repo/main/diary/$date.txt"
        val client = OkHttpClient()
        val request = Request.Builder().url(diaryUrl).build()

        client.newCall(request).enqueue(object : Callback {
            override fun onFailure(call: Call, e: IOException) {//github에 요청실패시
                runOnUiThread {
                    diaryContentTextView.text = "일기 내용이 없습니다."//해당날짜에 일기 없으면 일기 없다고 뜨기
                    Toast.makeText(this@Music_Main, "일기 가져오기 실패: ${e.message}", Toast.LENGTH_SHORT)
                        .show()
                }
            }

            override fun onResponse(call: Call, response: Response) {
                if (response.isSuccessful) {
                    val diaryText = response.body?.string()
                    runOnUiThread {
                        if (!diaryText.isNullOrEmpty()) {
                            diaryContentTextView.text = diaryText  // TextView에 일기 내용 표시
                        } else {
                            diaryContentTextView.text = "일기 내용이 없습니다."//해당날짜에 일기 없으면 일기 없다고 뜨기
                        }
                    }
                } else {
                    runOnUiThread {
                        diaryContentTextView.text = "일기 내용이 없습니다." //해당날짜에 일기 없으면 일기 없다고 뜨기
                        Toast.makeText(this@Music_Main, "일기 가져오기 실패", Toast.LENGTH_SHORT).show()
                    }
                }
            }
        })
    }

    // 켈린더에서 지정한 날짜에 있는 일기데이터로 음악 생성하게 하는함수
    private fun generateMusicFromDiary(diaryText: String) {
        if (!selectedDate.isNullOrEmpty()) {
            generateMusic(diaryText, selectedDate!!)
        } else {
            Toast.makeText(this, "먼저 날짜를 선택해 주세요.", Toast.LENGTH_SHORT).show()
        }
    }


    // 일기 텍스트를 기반으로 음악을 생성하는 함수
    private fun generateMusic(diaryText: String, date: String) {
        // 데이터 생성
        val client = OkHttpClient()
        val requestBody = FormBody.Builder()
            .add("text_prompt", diaryText) // 요청 본문에 diaryText 추가
            .add("selected_date", date)    // 요청 본문에 date 추가
            .build()

        val request = Request.Builder()
            .url("${BASE_URL}generate_music/") // 서버 URL
            .post(requestBody)                // POST 요청
            .build()

        // API 호출
        client.newCall(request).enqueue(object : Callback {
            override fun onFailure(call: Call, e: IOException) {
                runOnUiThread {
                    Toast.makeText(this@Music_Main, "음악 생성 요청 실패: ${e.message}", Toast.LENGTH_SHORT)
                        .show()
                }
            }

            override fun onResponse(call: Call, response: Response) {
                runOnUiThread {
                    if (response.isSuccessful) {
                        // 서버 응답 처리
                        val responseBody = response.body?.string()
                        Toast.makeText(
                            this@Music_Main,
                            "음악 생성 성공: $responseBody",
                            Toast.LENGTH_LONG
                        ).show()
                    } else {
                        Toast.makeText(
                            this@Music_Main,
                            "음악 생성 실패: ${response.message}",
                            Toast.LENGTH_SHORT
                        ).show()
                    }
                }
            }
        })
    }
}
//        // Retrofit 요청을 통해 Django 서버로 전송할 데이터 객체 생성
//        // 요청 생성
//        val request = Request.Builder()
//            .url(url)
//            .post(requestBody)
//            .build()
//        // Django 서버에 POST 요청 전송
//        apiService.generateMusic(request).enqueue(object : retrofit2.Callback<MusicResponse> {
//            override fun onResponse(call: retrofit2.Call<MusicResponse>, response: retrofit2.Response<MusicResponse>) {
//                if (response.isSuccessful && response.body() != null) {
//                    // 서버에서 반환된 음악 URL 받기
//                    val musicUrl = response.body()!!.music_url
//                    Toast.makeText(this@Music_Main, "Music URL: $musicUrl", Toast.LENGTH_LONG).show()
//                } else {
//                    Toast.makeText(this@Music_Main, "Failed to generate music", Toast.LENGTH_SHORT).show()
//                }
//            }
//
//            override fun onFailure(call: retrofit2.Call<MusicResponse>, t: Throwable) {
//                Toast.makeText(this@Music_Main, "Error: ${t.message}", Toast.LENGTH_SHORT).show()
//            }
//        })
//    }
//
//}