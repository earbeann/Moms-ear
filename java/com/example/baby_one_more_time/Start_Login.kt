package com.example.baby_one_more_time

import android.content.Intent
import android.os.Bundle
import android.util.Log
import android.widget.Button
import android.widget.EditText
import androidx.appcompat.app.AlertDialog
import androidx.appcompat.app.AppCompatActivity
import retrofit2.Call
import retrofit2.Callback
import retrofit2.Response
import retrofit2.Retrofit
import retrofit2.converter.gson.GsonConverterFactory

class Start_Login : AppCompatActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_login)

        // 뷰 참조
        val editText = findViewById<EditText>(R.id.editText)
        val editText2 = findViewById<EditText>(R.id.editText2)
        val button = findViewById<Button>(R.id.button)

        val retrofit= Retrofit.Builder()
            .baseUrl("http://192.168.101.93:8000/")
            .addConverterFactory(GsonConverterFactory.create())
            .build()

        val loginService = retrofit.create(LoginService::class.java)

        // Signup 버튼 참조
        val signupButton = findViewById<Button>(R.id.signup)

        // Signup 버튼 클릭 리스너
        signupButton.setOnClickListener {
            val intent = Intent(this, SignupActivity::class.java)
            startActivity(intent)
        }

        // 로그인 버튼 클릭 리스너
        button.setOnClickListener {
            val textID = editText.text.toString()
            val textPW = editText2.text.toString()

            loginService.requestLogin(textID, textPW).enqueue(object : Callback<Login> {
                override fun onFailure(p0: Call<Login>, t: Throwable) { // 네트워크 통신 실패
                    Log.d("DEBUG", t.message.toString())
                    val dialog = AlertDialog.Builder(this@Start_Login)
                        .setTitle("실패")
                        .setMessage("통신에 실패하였습니다.")
                        .setPositiveButton("확인", null)
                        .create()
                    dialog.show()
                }

                override fun onResponse(p0: Call<Login>, response: Response<Login>) { // 서버 응답 성공
                    if (response.isSuccessful) { // 응답 상태 코드가 200~299일 때
                        val login = response.body()
                        Log.d("LOGIN", "msg : " + login?.msg)
                        Log.d("LOGIN", "code : " + login?.code)
                        val dialog = AlertDialog.Builder(this@Start_Login)
                            .setTitle("알람")
                            .setMessage("${login?.msg}")
                            .setPositiveButton("확인", null)
                            .create()
                        dialog.show()
                        // Diary_Main으로 이동
                        val intent = Intent(this@Start_Login, Diary_Main::class.java)
                        startActivity(intent)
                        finish() // Start_Login 액티비티 종료
                    } else { // 응답 상태 코드가 400 이상일 때
                        val errorMsg = "회원정보가 올바르지 않습니다. 다시 로그인 해주세요."
                        Log.d("LOGIN", "Error: ${response.errorBody()?.string()}")
                        val dialog = AlertDialog.Builder(this@Start_Login)
                            .setTitle("로그인 실패")
                            .setMessage(errorMsg)
                            .setPositiveButton("확인", null)
                            .create()
                        dialog.show()
                    }
                }
            })
        }
    }
}
