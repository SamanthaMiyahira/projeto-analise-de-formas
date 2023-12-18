package com.example.projetomac447

import android.Manifest
import android.content.ContentResolver
import android.content.Context
import android.content.Intent
import android.content.pm.PackageManager
import android.net.Uri
import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import android.provider.OpenableColumns
import android.util.Log
import android.widget.Button
import android.widget.Toast
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import androidx.lifecycle.lifecycleScope
import com.google.android.gms.auth.api.signin.GoogleSignIn
import com.google.api.client.extensions.android.http.AndroidHttp
import com.google.api.client.googleapis.extensions.android.gms.auth.GoogleAccountCredential
import com.google.api.client.googleapis.extensions.android.gms.auth.UserRecoverableAuthIOException
import com.google.api.client.http.FileContent
import com.google.api.client.json.jackson2.JacksonFactory
import com.google.api.client.util.IOUtils
import com.google.api.services.drive.Drive
import com.google.api.services.drive.DriveScopes
import kotlinx.coroutines.*
import java.io.File
import java.io.FileInputStream
import java.io.FileOutputStream

class Storage : AppCompatActivity() {
    lateinit var mDrive: Drive
    private val MY_PERMISSIONS_REQUEST_WRITE_EXTERNAL_STORAGE = 123
    private val REQUEST_CODE_OPEN_DOCUMENT = 1

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_storage)

        // Check for WRITE_EXTERNAL_STORAGE permission
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.WRITE_EXTERNAL_STORAGE) != PackageManager.PERMISSION_GRANTED) {
            // Permission is not granted, request it
            ActivityCompat.requestPermissions(
                this,
                arrayOf(Manifest.permission.WRITE_EXTERNAL_STORAGE),
                MY_PERMISSIONS_REQUEST_WRITE_EXTERNAL_STORAGE
            )
            initializeUploadButton()
        } else {
            // Permission is already granted, continue with your code
            initializeUploadButton()
        }
    }

    private fun initializeUploadButton() {
        mDrive = getDriveService(this)
        var addAttachment = findViewById<Button>(R.id.upload)
        addAttachment.setOnClickListener {
            // Start the file picker activity
            val intent = Intent(Intent.ACTION_OPEN_DOCUMENT)
            intent.addCategory(Intent.CATEGORY_OPENABLE)
            intent.type = "image/*"
            startActivityForResult(intent, REQUEST_CODE_OPEN_DOCUMENT)
        }
    }

    // ... Other methods

    // Handle the result of the file picker activity
    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)
        println("entrou no onActivityResult")
        if (requestCode == REQUEST_CODE_OPEN_DOCUMENT && resultCode == RESULT_OK) {
            val selectedFile = data?.data // The Uri of the selected file
            // Handle the selected file as needed
            makeCopy(selectedFile!!)
            println(selectedFile.toString())
            uploadFileToGDrive(this, selectedFile.toString())
            Toast.makeText(this, selectedFile.toString(), Toast.LENGTH_LONG).show()
        }
    }
//
//    private fun initializeUploadButton() {
//        // Your existing code for initializing the upload button
//        mDrive = getDriveService(this)
//        var addAttachment = findViewById<Button>(R.id.upload)
//        addAttachment.setOnClickListener {
//            GlobalScope.async(Dispatchers.IO) {
////                val intent = Intent()
////                    .setType("*/*")
////                    .setAction(Intent.ACTION_GET_CONTENT)
////                startActivityForResult(Intent.createChooser(intent, "Select a file"), 111)
//                uploadFileToGDrive(this@Storage)
//                println("executou UPLOAD BUTTON")
//            }
//        }
//    }


    private fun getDriveService(context: Context): Drive {
        GoogleSignIn.getLastSignedInAccount(context).let { googleAccount ->
            val credential = GoogleAccountCredential.usingOAuth2(
                this, listOf(DriveScopes.DRIVE_FILE)
            )
            credential.selectedAccount = googleAccount!!.account!!
            return Drive.Builder(
                AndroidHttp.newCompatibleTransport(),
                JacksonFactory.getDefaultInstance(),
                credential
            )
                .setApplicationName(getString(R.string.app_name))
                .build()
        }
//        var tempDrive: Drive
//        return tempDrive
    }

    fun uploadFileToGDrive(context: Context, filepath: String) {
        mDrive.let { googleDriveService ->
            lifecycleScope.launch {
                try {
//                    val fileName = "Ticket"
                    val raunit = File(filepath)
                    raunit.setReadable(true)
                    raunit.setWritable(true)
                    println("raunit file coletado")
                    val gfile = com.google.api.services.drive.model.File()
                    gfile.name = "Subscribe"
                    val mimetype = "image/jpg"
                    val fileContent = FileContent(mimetype, raunit)
                    println("executou fileContent")
                    var fileid = ""

                    withContext(Dispatchers.Main) {
                        println("entrou no Dispatchers.Main")
                        withContext(Dispatchers.IO) {
                            println("entrou no Dispatchers.IO")
                            launch {
                                println("inicio execucao do launch do withContent")
                                var mFile =
                                    googleDriveService.Files().create(gfile, fileContent).execute()
                                println("entrou no launch do withContent")
                            }
                        }
                    }
                } catch (userAuthEx: UserRecoverableAuthIOException) {
                    startActivity(
                        userAuthEx.intent
                    )
                    println("entrou no catch do userAuthEx")
                } catch (e: Exception) {
                    e.printStackTrace()
                    Log.d("asdf", e.toString())
                    Toast.makeText(
                        context,
                        "Some Error Occured in Uploading Files" + e.toString(),
                        Toast.LENGTH_LONG
                    ).show()
                }

            }
        }
    }

//    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
//        super.onActivityResult(requestCode, resultCode, data)
//        if (requestCode == 111 && resultCode == RESULT_OK) {
//            val selectedFile = data!!.data //The uri with the location of the file
//            makeCopy(selectedFile!!)
//            Toast.makeText(this,selectedFile.toString(),Toast.LENGTH_LONG).show()
//        }
//    }

    private fun makeCopy(fileUri: Uri) {
        val parcelFileDescriptor = applicationContext.contentResolver.openFileDescriptor(fileUri, "r", null)
        val inputStream = FileInputStream(parcelFileDescriptor!!.fileDescriptor)
        val file = File(applicationContext.filesDir, getFileName(applicationContext.contentResolver, fileUri))
        val outputStream = FileOutputStream(file)
        IOUtils.copy(inputStream, outputStream)

    }

    private fun getFileName(contentResolver: ContentResolver, fileUri: Uri): String {

        var name = ""
        val returnCursor = contentResolver.query(fileUri, null, null, null, null)
        if (returnCursor != null) {
            val nameIndex = returnCursor.getColumnIndex(OpenableColumns.DISPLAY_NAME)
            returnCursor.moveToFirst()
            name = returnCursor.getString(nameIndex)
            returnCursor.close()
        }

        return name
    }

}