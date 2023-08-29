import click
from predict import predict, tts, img_input_show, img_dataset_show

@click.command()
@click.option('--aloud', default=False, help='output in speech form')
@click.option('--show-input', default=False, help='show the processed input image')
@click.option('--show-sample-data', default=False, help='show the processed input image')
def recog(aloud, show_input, show_sample_data):
    try:
        final_output = predict()
        click.echo("The recognized set of words are: " + final_output)
        if aloud:
            tts(final_output)
        if show_input:
            img_input_show()
        if show_sample_data:
            img_dataset_show()
        
    except KeyboardInterrupt:
        click.echo("Aborting the program...")
    
    except:
        click.echo("An error occured!")