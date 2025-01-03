package com.example.baby_one_more_time

import android.content.Context
import android.net.Uri
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.ImageView
import androidx.recyclerview.widget.RecyclerView
import com.bumptech.glide.Glide

// 이미지 뷰페이저 어댑터 클래스 정의
class FairyTale3Adapter(
    private val context: Context,   // Activity나 Fragment의 컨텍스트를 저장
    private val imageUris: List<Uri> // 이미지 URI 목록을 저장
) : RecyclerView.Adapter<FairyTale3Adapter.ImageViewHolder>() {

    // 뷰 홀더 클래스 정의, RecyclerView의 각 아이템 뷰를 관리
    inner class ImageViewHolder(itemView: View) : RecyclerView.ViewHolder(itemView) {
        val imageView: ImageView = itemView.findViewById(R.id.imageView) // 아이템 뷰에서 imageView를 찾음
    }

    override fun onCreateViewHolder(parent: ViewGroup, viewType: Int): ImageViewHolder {
        val view = LayoutInflater.from(context).inflate(R.layout.fairytale3, parent, false)
        return ImageViewHolder(view) // 변환한 뷰 객체를 담은 ViewHolder 반환
    }

    override fun onBindViewHolder(holder: ImageViewHolder, position: Int) {
        val imageUri = imageUris[position] // 현재 위치의 이미지 URI 가져옴
        Glide.with(context).load(imageUri).into(holder.imageView) // Glide를 사용해 이미지 로드
    }

    override fun getItemCount(): Int = imageUris.size // 이미지 URI 목록의 크기 반환
}
