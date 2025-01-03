package com.example.baby_one_more_time

import android.os.Bundle
import android.util.Log
import android.widget.Button
import android.widget.EditText
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import okhttp3.OkHttpClient
import okhttp3.logging.HttpLoggingInterceptor
import retrofit2.Call
import retrofit2.Callback
import retrofit2.Response
import retrofit2.Retrofit
import retrofit2.converter.gson.GsonConverterFactory

class SignupActivity : AppCompatActivity() {

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_signup)

        // Back 버튼 참조
        val backButton = findViewById<Button>(R.id.backButton)

        // Back 버튼 클릭 리스너
        backButton.setOnClickListener {
            finish() // 현재 액티비티 종료 -> 이전 화면으로 돌아감
        }

        // EditText와 Button 참조
        val nameInput = findViewById<EditText>(R.id.Input_name)
        val genderInput = findViewById<EditText>(R.id.Gender)
        val phoneInput = findViewById<EditText>(R.id.Input_PhoneNum)
        val birthdayInput = findViewById<EditText>(R.id.Input_Birthday)
        val emailInput = findViewById<EditText>(R.id.input_ID_1)
        val passwordInput = findViewById<EditText>(R.id.PW_input)
        val signupButton = findViewById<Button>(R.id.signup_button)

        // Retrofit 초기화
        val loggingInterceptor = HttpLoggingInterceptor().apply {
            level = HttpLoggingInterceptor.Level.BODY
        }
        val okHttpClient = OkHttpClient.Builder()
            .addInterceptor(loggingInterceptor)
            .build()

        val retrofit = Retrofit.Builder()
            .baseUrl("http://192.168.101.93:8000/") // Django 서버 주소
            .client(okHttpClient)
            .addConverterFactory(GsonConverterFactory.create())
            .build()

        val signupService = retrofit.create(SignupService::class.java)

        // 회원가입 버튼 클릭 리스너
        signupButton.setOnClickListener {
            val name = nameInput.text.toString()
            val gender = genderInput.text.toString()
            val phone = phoneInput.text.toString()
            val birthday = birthdayInput.text.toString()
            val email = emailInput.text.toString()
            val password = passwordInput.text.toString()

            // 입력값 검증
            if (name.isEmpty() || gender.isEmpty() || phone.isEmpty() || birthday.isEmpty() || email.isEmpty() || password.isEmpty()) {
                Toast.makeText(this, "모든 필드를 입력해주세요.", Toast.LENGTH_SHORT).show()
                return@setOnClickListener
            }

            // POST 요청
            val user = UserSignupRequest(name, gender, phone, birthday, email, password)
            signupService.signupUser(user).enqueue(object : Callback<UserSignupResponse> {
                override fun onResponse(
                    call: Call<UserSignupResponse>,
                    response: Response<UserSignupResponse>
                ) {
                    if (response.isSuccessful) {
                        Toast.makeText(this@SignupActivity, "회원가입 성공!", Toast.LENGTH_SHORT).show()
                        finish() // 회원가입 후 액티비티 종료
                    } else {
                        Toast.makeText(this@SignupActivity, "회원가입 실패: ${response.message()}", Toast.LENGTH_SHORT).show()
                        Log.d("SIGNUP", "Response: ${response.errorBody()?.string()}")
                    }
                }

                override fun onFailure(call: Call<UserSignupResponse>, t: Throwable) {
                    Toast.makeText(this@SignupActivity, "서버와의 통신 실패: ${t.message}", Toast.LENGTH_SHORT).show()
                    Log.e("SIGNUP", "Error: ${t.message}")
                }
            })
        }
    }
}
