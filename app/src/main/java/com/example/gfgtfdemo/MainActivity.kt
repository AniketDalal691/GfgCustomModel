package com.example.gfgtfdemo

import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import android.view.View
import android.widget.Button
import android.widget.EditText
import android.widget.TextView
import com.example.gfgtfdemo.ml.Iris
import org.tensorflow.lite.DataType
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer
import java.nio.ByteBuffer

class MainActivity : AppCompatActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)


        var ed1: EditText = findViewById(R.id.tf1);
        var ed2: EditText = findViewById(R.id.tf2);
        var ed3: EditText = findViewById(R.id.tf3);
        var ed4: EditText = findViewById(R.id.tf4);
        var txtView: TextView = findViewById(R.id.textView);
        var b: Button = findViewById<Button>(R.id.button);


        b.setOnClickListener(View.OnClickListener {
            val model = Iris.newInstance(this)

            var v1: Float = ed1.text.toString().toFloat();
            var v2: Float = ed2.text.toString().toFloat();
            var v3: Float = ed3.text.toString().toFloat();
            var v4: Float = ed4.text.toString().toFloat();


            var byte_buffer: ByteBuffer = ByteBuffer.allocateDirect(4 * 4)
            byte_buffer.putFloat(v1)
            byte_buffer.putFloat(v2)
            byte_buffer.putFloat(v3)
            byte_buffer.putFloat(v4)

            // Creates inputs for reference.
            val inputFeature0 = TensorBuffer.createFixedSize(intArrayOf(1, 4), DataType.FLOAT32)
            inputFeature0.loadBuffer(byte_buffer)

            // Runs model inference and gets result.
            val outputs = model.process(inputFeature0)
            val outputFeature0 = outputs.outputFeature0AsTensorBuffer.floatArray


            txtView.setText(
                "Iris-setosa : =" + outputFeature0[0].toString() + "\n" +
                "Iris-versicolor : =" + outputFeature0[1].toString() + "\n" +
                "Iris-virginica: =" +  outputFeature0[2].toString()
            )

            // Releases model resources if no longer used.
            model.close()

        })

    }
}