package com.example.projetomac447

import android.annotation.SuppressLint
import android.content.Intent
import android.graphics.Bitmap
import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import android.provider.MediaStore
import android.widget.Button
import android.widget.ImageView
import android.widget.TextView
import android.widget.Toast
import androidx.activity.result.ActivityResult
import androidx.activity.result.ActivityResultLauncher
import androidx.activity.result.contract.ActivityResultContracts
import com.chaquo.python.Python
import com.chaquo.python.android.AndroidPlatform
import com.google.firebase.auth.FirebaseAuth
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.GlobalScope
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.ByteArrayOutputStream

class MainActivity : AppCompatActivity() {
    private lateinit var auth: FirebaseAuth

    private val REQUEST_CODE = 22

    private lateinit var btnPicture: Button
    private lateinit var btnTrain: Button
    private lateinit var imageView: ImageView
    private var better_classifier: String = ""

    private lateinit var activityResultLauncher: ActivityResultLauncher<Intent>

    @SuppressLint("SetTextI18n")
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        // python file connection
        if (! Python.isStarted()){
            Python.start(AndroidPlatform(this));
        }

        val py = Python.getInstance()
        val module = py.getModule("imageClassification")

        btnPicture = findViewById(R.id.btncamera_id)
        btnTrain = findViewById(R.id.btntrain)
        imageView = findViewById(R.id.imageview1)

        var message = findViewById<TextView>(R.id.mensagem)

        var state = findViewById<TextView>(R.id.txtstate)


        btnTrain.setOnClickListener {
            print("executou Treinamento")
            state.text = "Please wait..."

            GlobalScope.launch(Dispatchers.IO) {
                val trainFunction = module["train_model"]

                // Call the Python function train_model() and get better_classifier
                val result = trainFunction?.call()?.toString()

                // Update the UI on the main thread
                withContext(Dispatchers.Main) {
                    better_classifier = result ?: ""
                    // Update the TextView with the result or any other logic you need
                    state.text = "Ready!"
                }
            }
        }

        activityResultLauncher =
            registerForActivityResult(ActivityResultContracts.StartActivityForResult()) { result: ActivityResult ->
                if (result.resultCode == RESULT_OK) {
                    println("chegou")
                    val photo: Bitmap? = result.data?.extras?.get("data") as Bitmap?
                    imageView.setImageBitmap(photo)
                    if (photo != null) {
                        val byteArrayOutputStream = ByteArrayOutputStream()
                        photo.compress(Bitmap.CompressFormat.PNG, 100, byteArrayOutputStream)
                        val byteArray = byteArrayOutputStream.toByteArray()

                        val funcao = module["recognize_image"]
                        val resultant = funcao?.call(byteArray, better_classifier)?.toString()
                        message.text = "The object is: $resultant"
                    }
                } else {
                    Toast.makeText(this, "Cancelled", Toast.LENGTH_SHORT).show()
                }
            }

        btnPicture.setOnClickListener {
            val cameraIntent = Intent(MediaStore.ACTION_IMAGE_CAPTURE)
            activityResultLauncher.launch(cameraIntent)
        }

        var storage=findViewById<Button>(R.id.storage)
        storage.setOnClickListener {
            startActivity(Intent(this,Storage::class.java))
        }
    }
}