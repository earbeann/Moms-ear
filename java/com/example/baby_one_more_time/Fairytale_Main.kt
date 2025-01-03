package com.example.baby_one_more_time

import android.content.Intent
import android.os.Bundle
import android.util.Log
import android.widget.Button
import android.widget.CalendarView
import android.widget.ImageButton
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.appcompat.widget.Toolbar
import okhttp3.*
import java.io.IOException
import android.net.Uri
import android.widget.TextView //일기데이터 미리보기 11.15


class Fairytale_Main : AppCompatActivity() {
    private lateinit var createFairyTaleButton: Button
    private lateinit var selectedDiaryContent: String
    private lateinit var selectedDate: String
    private lateinit var diaryContentTextView: TextView //일기데이터 미리보기 11.15
    private var isRequestInProgress = false  // 중복 요청 방지 플래그


    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.fairytale_main)
        createFairyTaleButton = findViewById(R.id.create_fairy_tale_button)
        diaryContentTextView = findViewById(R.id.diaryEntry1) //일기데이터 미리보기 11.15

        val toolbar: Toolbar = findViewById(R.id.toolbar)
        setSupportActionBar(toolbar)
        supportActionBar?.setDisplayShowTitleEnabled(false)
        toolbar.title = ""
        supportActionBar?.setDisplayHomeAsUpEnabled(true)
        toolbar.setNavigationOnClickListener { finish() }

        findViewById<ImageButton>(R.id.btn_account).setOnClickListener {
            // startActivity(Intent(this, AccountActivity::class.java))
        }


        // CalendarView 설정 및 날짜 선택 리스너 추가
        val calendarView = findViewById<CalendarView>(R.id.calendarView)
        calendarView.setOnDateChangeListener { _, year, month, dayOfMonth ->
            // 날짜 선택 시 전단계로 돌아가지 않도록 실행 순서를 조정
            selectedDate = String.format("%04d-%02d-%02d", year, month + 1, dayOfMonth)
//            fetchDiaryEntry(selectedDate)
            fetchDiaryEntryAndGenerateFairyTale(selectedDate)
        }

        // 동화 읽기 버튼 클릭 시 이미지를 로드하고, 성공 시 FairyTale2Activity로 이동
        findViewById<Button>(R.id.Read_fairy_tale_button).setOnClickListener {
            if (::selectedDate.isInitialized) { //중복 오류 수정 11.18
                loadImagesFromServer(selectedDate)
            } else {
                Toast.makeText(this, "날짜를 선택해 주세요.", Toast.LENGTH_SHORT).show()
            }
        }

        // 위에 동화생성 버튼을 누르면 동화 생성되는거 일기 데이터 가져오게 하려고 이렇게 바꿈!!!!!!!!!!!!!!!!!!!!!!!!!!
        val createFairyTaleButton = findViewById<Button>(R.id.create_fairy_tale_button)
        createFairyTaleButton.setOnClickListener {
            //동화생성 중복오류 수정중 11.18
            if (isRequestInProgress) {
                Toast.makeText(this, "이미 요청 중입니다. 잠시만 기다려 주세요.", Toast.LENGTH_SHORT).show()
                return@setOnClickListener
            }
            if (::selectedDiaryContent.isInitialized) { // selectedDiaryContent가 초기화된 경우에만 호출
                //동화생성 중복오류 수정중 11.18
                isRequestInProgress = true
                createFairyTaleButton.isEnabled = false

                Log.d("FairyTale_Main", "Selected Diary Content: $selectedDiaryContent") // 11-14 16:18 방금 추가한거!!!!!!!!!!!!!
                sendFairyTaleCreationRequest(selectedDiaryContent, selectedDate)
            } else {
                Toast.makeText(this, "일기 내용을 먼저 선택하세요.", Toast.LENGTH_SHORT).show()
            }
        }
        setBottomNavigation()
    }
    private fun setBottomNavigation() {
        findViewById<Button>(R.id.fairyTaleButton).setOnClickListener {
            startActivity(Intent(this, Fairytale_Main::class.java))
        }

        findViewById<Button>(R.id.homeButton).setOnClickListener {
            startActivity(Intent(this, Diary_Main::class.java))
        }

        findViewById<Button>(R.id.musicButton).setOnClickListener {
            startActivity(Intent(this, Music_Main::class.java))
        }

        findViewById<Button>(R.id.myPageButton).setOnClickListener {
            startActivity(Intent(this, MyPage_Main::class.java))
        }
    }

    //새로 만든 켈린더 지정 날짜에 동화 생성
    private fun fetchDiaryEntryAndGenerateFairyTale(date: String) {
        val client = OkHttpClient()
        val diaryUrl = "https://raw.githubusercontent.com/earbeann/mom-s_ear_repo/main/diary/$date.txt" // GitHub 일기 파일 경로
        val request = Request.Builder().url(diaryUrl).header("Cache-Control", "no-cache").build()

        client.newCall(request).enqueue(object : Callback {
            override fun onFailure(call: Call, e: IOException) {
                runOnUiThread {
                    Toast.makeText(this@Fairytale_Main, "일기 데이터를 불러오지 못했습니다.", Toast.LENGTH_SHORT).show()
                }
            }

            override fun onResponse(call: Call, response: Response) {
                if (response.isSuccessful) {
                    response.body?.let { responseBody ->
                        val content = responseBody.string()
                        runOnUiThread {
                            selectedDiaryContent = content
                            diaryContentTextView.text = content
                            Toast.makeText(this@Fairytale_Main, "일기를 불러왔습니다. 동화를 생성합니다.", Toast.LENGTH_SHORT).show()
//                            sendFairyTaleCreationRequest(selectedDiaryContent, date) // 일기 내용과 날짜로 동화 생성 요청
                        }
                    }
                } else {
                    runOnUiThread {
                        Toast.makeText(this@Fairytale_Main, "해당 날짜에 일기가 없습니다.", Toast.LENGTH_SHORT).show()
                    }
                }
            }
        })
    }


    private fun sendFairyTaleCreationRequest(selectedDiaryContent: String, date: String) {
        val client = OkHttpClient()
        val url = "http://192.168.101.93:8000/create_story/" // Django 서버 URL로 변경
        val requestBody = FormBody.Builder()
            .add("diary_content", selectedDiaryContent) // 일기 내용을 여기에 넣어 전송
            .add("date", date) // 날짜 추가 11.17
            .build()

        val request = Request.Builder()
            .url(url)
            .post(requestBody)
            .header("Cache-Control", "no-cache")  // 캐시 방지 헤더 추가
            .build()

        client.newCall(request).enqueue(object : Callback {

            override fun onFailure(call: Call, e: IOException) {
                runOnUiThread {
                    Toast.makeText(this@Fairytale_Main, "동화 생성 대기중", Toast.LENGTH_SHORT).show()
                    isRequestInProgress = false // 요청 상태 해제
                    //동화생성 중복오류 수정중 11.18
                    createFairyTaleButton.isEnabled = true
                }

            }

            override fun onResponse(call: Call, response: Response) {
                runOnUiThread {
                    //동화생성 중복오류 수정중 11.18
                    isRequestInProgress = false
                    createFairyTaleButton.isEnabled = true
                    if (response.isSuccessful) {
                        try {
                            val intent = Intent(this@Fairytale_Main, FairyTale2Activity::class.java)
                            startActivity(intent)
                        } catch (e: Exception) {
                            Log.e("FairyTale_Main", "Activity 전환 중 오류: ${e.message}")
                            Toast.makeText(this@Fairytale_Main, "페이지 전환 중 오류가 발생했습니다.", Toast.LENGTH_SHORT).show()
                        }
                    } else {
                        val code = response.code
                        val message = response.message
                        Toast.makeText(this@Fairytale_Main, "동화 생성 실패2: $code, $message", Toast.LENGTH_SHORT).show()

                    }
                }
            }
        })
    }


    // 서버로부터 날짜별 이미지를 불러오는 함수 !!!!!!!!!!!!!!! 이 이하는 종현 추가
    private fun loadImagesFromServer(dateFolder: String) {
        val client = OkHttpClient()
        val baseUrl = "https://raw.githubusercontent.com/earbeann/mom-s_ear_repo/main/fairytale/$dateFolder"
        val imageUris = ArrayList<Uri>()
        var index = 1

        // 이미지가 존재하지 않을 때까지 반복하여 URL을 확인
        fun fetchImageUrls() {
            val imageUrl = "$baseUrl/image_$index.png?timestamp=${System.currentTimeMillis()}"
            val request = Request.Builder()
                .url(imageUrl)
                .header("Cache-Control", "no-store")  // 캐시 무효화 헤더 추가
                .get() // GET 요청으로 URL 유효성 확인
                .build()

            client.newCall(request).enqueue(object : Callback {
                override fun onFailure(call: Call, e: IOException) {
                    runOnUiThread {

                        Toast.makeText(this@Fairytale_Main, "Failed to load images", Toast.LENGTH_SHORT).show()
                    }
                }

                override fun onResponse(call: Call, response: Response) {
                    response.use {
                        if (response.isSuccessful) {
                            imageUris.add(Uri.parse(imageUrl))
                            index++
                            fetchImageUrls() // 다음 인덱스로 재귀 호출
                        } else {
                            runOnUiThread {
                                if (imageUris.isNotEmpty()) {
                                    startFairyTale2Activity(imageUris)
                                } else {
                                    Toast.makeText(this@Fairytale_Main, "No images found for this date", Toast.LENGTH_SHORT).show()
                                }
                            }
                        }
                    }
                }
            })
        }

        // 첫 번째 이미지 URL 확인 시작
        fetchImageUrls()
    }
    private fun startFairyTale2Activity(imageUris: ArrayList<Uri>) {
        val intent = Intent(this, FairyTale2Activity::class.java)
        intent.putParcelableArrayListExtra("imageUris", imageUris)
        startActivity(intent)
    }
}
