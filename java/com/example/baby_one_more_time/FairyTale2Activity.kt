package com.example.baby_one_more_time

import android.content.Intent
import android.os.Bundle
import androidx.activity.enableEdgeToEdge
import androidx.appcompat.app.AppCompatActivity
import androidx.core.view.ViewCompat
import androidx.core.view.WindowInsetsCompat
import androidx.viewpager2.widget.ViewPager2
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.Button
import android.widget.ImageView
import android.widget.ImageButton
import androidx.recyclerview.widget.RecyclerView
import android.net.Uri

class FairyTale2Activity : AppCompatActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        setContentView(R.layout.fairytale2)

        // 시스템 바 여백 처리
        ViewCompat.setOnApplyWindowInsetsListener(findViewById(R.id.main)) { v, insets ->
            val systemBars = insets.getInsets(WindowInsetsCompat.Type.systemBars())
            v.setPadding(systemBars.left, systemBars.top, systemBars.right, systemBars.bottom)
            insets
        }

        // 전달받은 이미지 URI 리스트
        val imageUris = intent.getParcelableArrayListExtra<Uri>("imageUris") ?: arrayListOf()

        // ViewPager2 설정
        val viewPager: ViewPager2 = findViewById(R.id.viewPager2)

        // 어댑터 설정
        val adapter = FairyTale3Adapter(this, imageUris)
        viewPager.adapter = adapter

        // 뒤로 가기 버튼 클릭 리스너 추가
        val backButton = findViewById<ImageButton>(R.id.btn_back)
        backButton.setOnClickListener {
            // MainActivity로 돌아가기
            val intent = Intent(this, Fairytale_Main::class.java)
            startActivity(intent)
            finish() // 현재 Activity 종료
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
        // 하단바 설정
        setupBottomNavigation()    // setupBottomNavigation 함수 내가 만든거 방금 11-14 18:40분!!!!!!

    }
    // setupBottomNavigation 함수 내가 만든거 방금 11-14 18:40분!!!!!!
    private fun setupBottomNavigation() {
        val fairyTaleButton = findViewById<Button>(R.id.fairyTaleButton)
        fairyTaleButton.setOnClickListener {
            startActivity(Intent(this, Fairytale_Main::class.java))
        }
        val homeButton = findViewById<Button>(R.id.homeButton)
        homeButton.setOnClickListener {
            startActivity(Intent(this, Diary_Main::class.java))
        }
        val musicButton = findViewById<Button>(R.id.musicButton)
        musicButton.setOnClickListener {
            startActivity(Intent(this, Music_Main::class.java))
        }
        val myButton = findViewById<Button>(R.id.myPageButton)
        myButton.setOnClickListener {
            startActivity(Intent(this, MyPage_Main::class.java))
        }
    }
}

class ImagePagerAdapter(private val imageUris: List<Uri>) : RecyclerView.Adapter<ImagePagerAdapter.ImageViewHolder>() {

    inner class ImageViewHolder(itemView: View) : RecyclerView.ViewHolder(itemView) {
        val imageView: ImageView = itemView.findViewById(R.id.imageView)
    }

    override fun onCreateViewHolder(parent: ViewGroup, viewType: Int): ImageViewHolder {
        val view = LayoutInflater.from(parent.context).inflate(R.layout.fairytale3, parent, false)
        return ImageViewHolder(view)
    }

    override fun onBindViewHolder(holder: ImageViewHolder, position: Int) {
        val imageUri = imageUris[position]
        holder.imageView.setImageURI(imageUri) // 이미지 URI로 이미지 설정
    }

    override fun getItemCount(): Int = imageUris.size
}
//class ImagePagerAdapter(private val images: List<Int>) : RecyclerView.Adapter<ImagePagerAdapter.ImageViewHolder>() {
//
//    inner class ImageViewHolder(itemView: View) : RecyclerView.ViewHolder(itemView) {
//        val imageView: ImageView = itemView.findViewById(R.id.imageView)
//    }
//
//    override fun onCreateViewHolder(parent: ViewGroup, viewType: Int): ImageViewHolder {
//        val view = LayoutInflater.from(parent.context).inflate(R.layout.fairytale3, parent, false)
//        return ImageViewHolder(view)
//    }
//
//    override fun onBindViewHolder(holder: ImageViewHolder, position: Int) {
//        holder.imageView.setImageResource(images[position])
//    }
//
//    override fun getItemCount(): Int = images.size
//}
