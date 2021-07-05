let net;

const imgEl = document.getElementById('img');
const descEl = document.getElementById('descripcion_imagen');

async function app(){

    net = await mobilenet.load();

    var result = await net.classify(imgEl);
    console.log(result);
}

app();